import torch.nn as nn

from losses.aam_softmax import AAMSoftmaxLoss
from losses.infonce import SupervisedMaskedInfoNCELoss
from losses.noise import NoiseReconstructionMSELoss


class NoiseLoRASVLoss(nn.Module):
    def __init__(self, cfg, embedding_dim=192):
        super().__init__()
        self.speaker_weight = float(cfg.get("speaker_weight", 1.0))
        self.distill_weight = float(cfg.get("distill_weight", 1.0))
        self.noise_weight = float(cfg.get("noise_weight", 0.1))
        self.speaker = AAMSoftmaxLoss(
            cfg.get("n_classes", 1211),
            embedding_dim=embedding_dim,
            margin=cfg.get("aam_margin", 0.2),
            scale=cfg.get("aam_scale", 30),
        )
        self.distill = SupervisedMaskedInfoNCELoss(cfg.get("infonce_temperature", 0.07))
        self.noise = NoiseReconstructionMSELoss()

    def forward(self, outputs, labels, noise_target_logmel=None, noise_valid=None):
        embedding = outputs["embedding"]
        speaker_loss, acc, logits = self.speaker(embedding, labels)
        if self.distill_weight > 0:
            if outputs.get("teacher_embedding") is None:
                raise ValueError("teacher_embedding is required when distill_weight > 0")
            # Eq. (6): use the full batch and mask false negatives from the same speaker.
            distill_loss = self.distill(embedding, outputs.get("teacher_embedding"), labels)
        else:
            distill_loss = embedding.new_tensor(0.0)
        if self.noise_weight > 0:
            if outputs.get("noise_hat") is None or noise_target_logmel is None:
                raise ValueError("noise_hat and noise_target_logmel are required when noise_weight > 0")
            noise_loss = self.noise(outputs.get("noise_hat"), noise_target_logmel, noise_valid)
        else:
            noise_loss = embedding.new_tensor(0.0)
        total = (
            self.speaker_weight * speaker_loss
            + self.distill_weight * distill_loss
            + self.noise_weight * noise_loss
        )
        return {
            "loss": total,
            "speaker_loss": speaker_loss.detach(),
            "distill_loss": distill_loss.detach(),
            "noise_loss": noise_loss.detach(),
            "top1": acc.detach(),
            "logits": logits,
        }
