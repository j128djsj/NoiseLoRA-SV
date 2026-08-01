import torch.nn as nn


class NoiseReconstructionMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, pred_logmel, target_logmel, valid=None):
        if pred_logmel is None or target_logmel is None:
            raise ValueError("noise reconstruction and target are required when noise loss is enabled")
        frames = min(pred_logmel.size(-1), target_logmel.size(-1))
        pred = pred_logmel[..., :frames]
        target = target_logmel[..., :frames].to(pred.device)
        loss = self.criterion(pred, target).mean(dim=(1, 2))
        if valid is not None:
            valid = valid.to(pred.device).float()
            denom = valid.sum().clamp_min(1.0)
            return (loss * valid).sum() / denom
        return loss.mean()
