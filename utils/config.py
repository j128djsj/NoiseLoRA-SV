import os

import yaml


def validate_config(cfg):
    if int(cfg.get("audio", {}).get("n_mels", 80)) != 80:
        # Keep the frontend dimension aligned with the paper model.
        raise ValueError("NoiseLoRA-SV currently requires audio.n_mels=80 to match the paper model.")
    model_cfg = cfg.get("model", {})
    opt_cfg = cfg.get("optimizer", {})
    paths = cfg.get("paths", {})
    if not model_cfg.get("use_noiselora", False) and float(opt_cfg.get("speaker_lr_scale", 1.0)) < 1.0:
        if not paths.get("student_checkpoint") and not paths.get("teacher_checkpoint"):
            import warnings

            warnings.warn(
                "Randomly initialized baseline ECAPA should use optimizer.speaker_lr_scale=1.0",
                RuntimeWarning,
                stacklevel=2,
            )
    return cfg


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg["_config_dir"] = os.path.dirname(os.path.abspath(path))
    return validate_config(cfg)
