import os
from copy import deepcopy

import numpy as np
import torch

from noiselora_sv.training.data import build_eval_loader, build_train_loader
from noiselora_sv.training.logger import build_logger
from noiselora_sv.losses import NoiseLoRASVLoss
from noiselora_sv.models import build_model
from noiselora_sv.training.optimizer import build_optimizer
from noiselora_sv.training.scheduler import build_scheduler
from noiselora_sv.utils.checkpoint import (
    NOISELORA_REQUIRED_PREFIXES,
    load_complete_baseline_checkpoint,
    load_full_model_checkpoint,
    load_module_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from noiselora_sv.utils.config import validate_config
from noiselora_sv.utils.metrics import compute_eer, cosine_score
from noiselora_sv.utils.model_summary import format_parameter_summary, summarize_model_parameters
from noiselora_sv.utils.seed import set_seed


class NoiseLoRASVTask:
    def __init__(self, cfg, mode="train", resume="", pretrained_eval=False):
        validate_config(cfg)
        self.cfg = deepcopy(cfg)
        self.mode = mode
        self.pretrained_eval = bool(pretrained_eval)
        set_seed(cfg.get("seed", 0))
        self.device = self._device(cfg.get("device", "auto"))
        model_cfg = self.cfg.setdefault("model", {})
        if self.mode == "eval":
            model_cfg["use_teacher"] = False
        self.model = build_model(self.cfg).to(self.device)
        self.logger = build_logger(cfg.get("paths", {}))
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.use_amp = False
        self.scaler = None
        self.start_epoch = 0
        self._eval_checkpoint_loaded = False
        # Training owns loss and optimizer state; evaluation only owns inference state.
        if self.mode == "train":
            emb_dim = cfg.get("model", {}).get("embedding_dim", 192)
            self.criterion = NoiseLoRASVLoss(cfg.get("loss", {}), embedding_dim=emb_dim).to(self.device)
            self.optimizer = build_optimizer(self.model, self.criterion, cfg.get("optimizer", {}))
            self.scheduler = build_scheduler(self.optimizer, cfg.get("scheduler", {}))
            self.use_amp = bool(cfg.get("training", {}).get("amp", False)) and self.device.type == "cuda"
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
            if resume:
                state = load_training_checkpoint(resume, self.model, self.criterion, self.optimizer, self.scheduler, scaler=self.scaler)
                self.start_epoch = int(state["epoch"])
            else:
                self._load_training_initial_checkpoints()
        elif self.mode == "eval" and self.pretrained_eval:
            self._load_eval_pretrained()
        elif self.mode == "eval":
            pass
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        self.parameter_summary = summarize_model_parameters(self.model)
        print(format_parameter_summary(self.parameter_summary))

    @staticmethod
    def _device(name):
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)

    def _encoder_target(self):
        return self.model.student if hasattr(self.model, "student") else self.model.encoder

    def _load_training_initial_checkpoints(self):
        paths = self.cfg.get("paths", {})
        student_ckpt = paths.get("student_checkpoint", "")
        teacher_ckpt = paths.get("teacher_checkpoint", "")
        use_noiselora = bool(self.cfg.get("model", {}).get("use_noiselora", False))
        distill_weight = float(self.cfg.get("loss", {}).get("distill_weight", 0.0))
        if distill_weight > 0 and hasattr(self.model, "teacher") and self.model.teacher is not None and not teacher_ckpt:
            raise RuntimeError("loss.distill_weight > 0 requires paths.teacher_checkpoint for a frozen clean teacher")
        student_source = student_ckpt or teacher_ckpt
        if use_noiselora and not student_source:
            raise RuntimeError("NoiseLoRA-SV training requires paths.student_checkpoint or paths.teacher_checkpoint")
        if student_source:
            source_name = "student_checkpoint" if student_ckpt else "teacher_checkpoint"
            # Start the student from a clean ECAPA checkpoint.
            print(f"Initializing student from {source_name}: {student_source}")
            load_module_checkpoint(
                self._encoder_target(),
                student_source,
                prefixes=("student.", "encoder.", "speaker_encoder."),
                label="student",
            )
        if teacher_ckpt and hasattr(self.model, "teacher") and self.model.teacher is not None:
            print(f"Initializing frozen teacher from teacher_checkpoint: {teacher_ckpt}")
            load_module_checkpoint(
                self.model.teacher,
                teacher_ckpt,
                prefixes=("teacher.", "student.", "encoder.", "speaker_encoder."),
                label="teacher",
            )

    def _load_eval_pretrained(self):
        if self.cfg.get("model", {}).get("use_noiselora", False):
            raise RuntimeError("--pretrained-eval is only supported by the baseline ECAPA configuration")
        paths = self.cfg.get("paths", {})
        source = paths.get("student_checkpoint", "") or paths.get("teacher_checkpoint", "")
        if not source:
            raise RuntimeError("--pretrained-eval requires paths.student_checkpoint or paths.teacher_checkpoint")
        print(f"Initializing evaluation encoder from pretrained checkpoint: {source}")
        load_module_checkpoint(
            self._encoder_target(),
            source,
            prefixes=("student.", "encoder.", "speaker_encoder."),
            label="eval_pretrained",
        )

    def _noise_target_logmel(self, batch):
        if "noise_target" not in batch:
            return None
        with torch.no_grad():
            return self.model.frontend(batch["noise_target"].to(self.device), aug=False)

    def train(self):
        if self.mode != "train":
            raise RuntimeError("train() is only available in training mode")
        loader = build_train_loader(self.cfg)
        epochs = int(self.cfg.get("training", {}).get("epochs", 200))
        ckpt_dir = self.cfg.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        for epoch in range(self.start_epoch + 1, epochs + 1):
            if hasattr(loader.batch_sampler, "set_epoch"):
                loader.batch_sampler.set_epoch(epoch)
            stats = self.train_epoch(loader, epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            self.logger.info("epoch", epoch=epoch, **stats)
            save_training_checkpoint(
                os.path.join(ckpt_dir, "last.pth"),
                self.model,
                self.criterion,
                self.optimizer,
                self.scheduler,
                epoch,
                cfg=self.cfg,
                scaler=self.scaler,
            )

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss, total_top1, total_count = 0.0, 0.0, 0
        for batch in loader:
            noisy = batch["noisy"].to(self.device)
            clean = batch["clean"].to(self.device)
            labels = batch["label"].to(self.device)
            valid = batch.get("noise_valid")
            valid = valid.to(self.device) if valid is not None else None
            self.optimizer.zero_grad(set_to_none=True)
            # Keep AMP CUDA-only and controlled by YAML.
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(noisy, clean=clean, aug=True)
                loss_pack = self.criterion(outputs, labels, self._noise_target_logmel(batch), valid)
            self.scaler.scale(loss_pack["loss"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            bsz = int(labels.size(0))
            total_loss += float(loss_pack["loss"].detach().cpu()) * bsz
            total_top1 += float(loss_pack["top1"].detach().cpu()) * bsz
            total_count += bsz
        denom = max(total_count, 1)
        # Keep epoch metadata separate from scalar training statistics.
        return {"loss": total_loss / denom, "top1": total_top1 / denom}

    @torch.no_grad()
    def evaluate(self, condition="clean", snr=None, checkpoint="", noise_type=None):
        self._validate_eval_request(condition, snr, noise_type)
        if checkpoint:
            if self.cfg.get("model", {}).get("use_noiselora", False):
                load_full_model_checkpoint(
                    self.model,
                    checkpoint,
                    required_prefixes=NOISELORA_REQUIRED_PREFIXES,
                    min_loaded_percent=90.0,
                    label="noiselora_eval",
                )
            else:
                load_complete_baseline_checkpoint(self.model, checkpoint, strict=False)
            self._eval_checkpoint_loaded = True
        if not self._eval_checkpoint_loaded and not self.pretrained_eval:
            raise RuntimeError("Evaluation requires --checkpoint unless --pretrained-eval is set")
        loader, dataset = build_eval_loader(self.cfg, condition=condition, snr=snr, noise_type=noise_type)
        self.model.eval()
        embeddings = {}
        for wav, rel_path in loader:
            emb = self.model.extract_embedding(wav.to(self.device))
            for path, vector in zip(rel_path, emb.detach().cpu().numpy()):
                embeddings[str(path)] = vector
        scores, labels, skipped = [], [], 0
        for label, first, second in dataset.trials:
            if first in embeddings and second in embeddings:
                scores.append(cosine_score(embeddings[first], embeddings[second]))
                labels.append(label)
            else:
                skipped += 1
        if not scores:
            raise RuntimeError("No verification trials were scored")
        positives = int(np.sum(np.asarray(labels) == 1))
        negatives = int(np.sum(np.asarray(labels) == 0))
        if positives == 0 or negatives == 0:
            raise RuntimeError("Evaluation requires both positive and negative trials")
        eer = compute_eer(np.asarray(scores), np.asarray(labels))
        result = {
            "condition": condition,
            "noise_type": noise_type,
            "snr": snr,
            "eer": eer,
            "trials": len(scores),
            "positive_trials": positives,
            "negative_trials": negatives,
            "skipped_trials": skipped,
            "total_trials": len(dataset.trials),
        }
        self.logger.info("eval", **result)
        return result

    @staticmethod
    def _validate_eval_request(condition, snr, noise_type):
        if condition not in {"clean", "seen", "unseen"}:
            raise ValueError(f"Unsupported evaluation condition: {condition}")
        if condition == "clean" and (snr is not None or noise_type is not None):
            raise ValueError("clean evaluation does not accept snr or noise_type")
        if condition == "seen":
            if noise_type not in {"babble", "music", "noise"}:
                raise ValueError("seen evaluation requires noise_type in {babble, music, noise}")
            if snr is None:
                raise ValueError("seen evaluation requires snr")
        if condition == "unseen":
            if noise_type is not None:
                raise ValueError("unseen evaluation does not accept noise_type")
            if snr is None:
                raise ValueError("unseen evaluation requires snr")
