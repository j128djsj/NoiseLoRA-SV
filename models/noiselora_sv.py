import torch
import torch.nn as nn

from models.audio_frontend import LogMelFrontend
from models.crn import CRNNoiseReconstructionNetwork
from models.ecapa_tdnn import ECAPATDNN
from models.noise_lora import GlobalNoiseConditionedLoRA, HNCNoiseConditionedLoRA


class BaselineECAPA(nn.Module):
    def __init__(self, audio_cfg, model_cfg):
        super().__init__()
        self.frontend = LogMelFrontend(audio_cfg)
        self.encoder = ECAPATDNN(model_cfg.get("channels", 1024), model_cfg.get("embedding_dim", 192))

    def forward(self, wav, clean=None, aug=False):
        logmel = self.frontend(wav, aug=aug)
        emb = self.encoder.forward_logmel(logmel)
        return {"embedding": emb}

    def extract_embedding(self, wav):
        return self.forward(wav, aug=False)["embedding"]


class NoiseLoRASV(nn.Module):
    def __init__(self, audio_cfg, model_cfg):
        super().__init__()
        channels = int(model_cfg.get("channels", 1024))
        embedding_dim = int(model_cfg.get("embedding_dim", 192))
        noise_dim = int(model_cfg.get("noise_dim", 128))
        crn_cfg = model_cfg.get("crn", {})
        lora_cfg = model_cfg.get("lora", {})
        self.frontend = LogMelFrontend(audio_cfg)
        self.noise_network = CRNNoiseReconstructionNetwork(
            n_mels=audio_cfg.get("n_mels", 80),
            base_channels=crn_cfg.get("base_channels", 32),
            rnn_channels=crn_cfg.get("rnn_channels", 64),
            rnn_hidden=crn_cfg.get("rnn_hidden", 256),
            noise_dim=noise_dim,
        )
        self.student = ECAPATDNN(channels, embedding_dim)
        self.teacher = ECAPATDNN(channels, embedding_dim) if model_cfg.get("use_teacher", True) else None
        if self.teacher is not None:
            for param in self.teacher.parameters():
                param.requires_grad = False
            # Keep the clean teacher frozen during training.
            self.teacher.eval()
        rank = lora_cfg.get("rank", 4)
        alpha = lora_cfg.get("alpha", 8)
        hidden = lora_cfg.get("hyper_hidden", 256)
        init_scale = lora_cfg.get("init_scale", 0.001)
        gate_hidden = lora_cfg.get("gate_hidden", 64)
        self.adapters = nn.ModuleDict({
            "C": GlobalNoiseConditionedLoRA(channels, noise_dim, rank, alpha, hidden, init_scale),
            "S1": GlobalNoiseConditionedLoRA(channels, noise_dim, rank, alpha, hidden, init_scale),
            "S2": HNCNoiseConditionedLoRA(channels, noise_dim, rank, alpha, hidden, gate_hidden, init_scale),
            "S3": HNCNoiseConditionedLoRA(channels, noise_dim, rank, alpha, hidden, gate_hidden, init_scale),
        })

    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def forward(self, noisy, clean=None, aug=False):
        noisy_logmel = self.frontend(noisy, aug=aug)
        noise_out = self.noise_network(noisy_logmel)
        z_global = noise_out["z_global"]
        z_local = noise_out["z_local"]
        # Adapt only the noisy student path with noise-conditioned LoRA.
        embedding = self.student.forward_logmel(noisy_logmel, self.adapters, z_global, z_local)
        teacher_embedding = None
        if self.teacher is not None and clean is not None:
            with torch.no_grad():
                clean_logmel = self.frontend(clean, aug=False)
                teacher_embedding = self.teacher.forward_logmel(clean_logmel)
        return {
            "embedding": embedding,
            "teacher_embedding": teacher_embedding,
            "noise_hat": noise_out["noise_hat"],
            "z_global": z_global,
            "z_local": z_local,
            "temporal_gates": {
                "S2": self.adapters["S2"].last_gate,
                "S3": self.adapters["S3"].last_gate,
            },
        }

    def extract_embedding(self, wav):
        return self.forward(wav, clean=None, aug=False)["embedding"]


def build_model(cfg):
    model_cfg = cfg.get("model", {})
    if model_cfg.get("use_noiselora", False):
        return NoiseLoRASV(cfg.get("audio", {}), model_cfg)
    return BaselineECAPA(cfg.get("audio", {}), model_cfg)
