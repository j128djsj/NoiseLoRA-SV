import torch


def build_scheduler(optimizer, cfg=None):
    cfg = cfg or {}
    name = str(cfg.get("name", "exponential")).lower()
    if name in ("none", "null"):
        return None
    if name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.get("gamma", 0.97)))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 1)),
            gamma=float(cfg.get("gamma", 0.97)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")
