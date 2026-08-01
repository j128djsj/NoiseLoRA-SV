import torch


def _named_trainable(module, prefix):
    if module is None:
        return []
    return [(f"{prefix}.{name}", param) for name, param in module.named_parameters() if param.requires_grad]


def build_optimizer(model, criterion=None, cfg=None):
    cfg = cfg or {}
    lr = float(cfg.get("lr", 0.001))
    weight_decay = float(cfg.get("weight_decay", 2e-5))
    speaker_lr_scale = float(cfg.get("speaker_lr_scale", 0.1))
    named = _named_trainable(model, "model") + _named_trainable(criterion, "loss")
    speaker_params, other_params = [], []
    for name, param in named:
        if ".student." in name or ".encoder." in name:
            speaker_params.append(param)
        else:
            other_params.append(param)
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": lr})
    if speaker_params:
        groups.append({"params": speaker_params, "lr": lr * speaker_lr_scale})
    if not groups:
        raise ValueError("No trainable parameters found")
    name = str(cfg.get("name", "adam")).lower()
    if name != "adam":
        raise ValueError(f"Unsupported optimizer: {name}")
    return torch.optim.Adam(groups, lr=lr, weight_decay=weight_decay)
