import os
from copy import deepcopy

import torch


LEGACY_ECAPA_PREFIXES = (
    ("speaker_encoder.conv1.", "C.0."),
    ("speaker_encoder.bn1.", "C.2."),
    ("speaker_encoder.layer1.", "S1."),
    ("speaker_encoder.layer2.", "S2."),
    ("speaker_encoder.layer3.", "S3."),
    ("speaker_encoder.layer4.", "mfa.0."),
    ("speaker_encoder.attention.", "pool.attention."),
    ("speaker_encoder.bn5.", "bn."),
    ("speaker_encoder.fc6.", "linear."),
    ("speaker_encoder.bn6.", "out_bn."),
)
IGNORED_LEGACY_PREFIXES = ("speaker_encoder.torchfbank.", "speaker_encoder.specaug.")
NOISELORA_REQUIRED_MODULES = (
    ("student", "student."),
    ("noise_network", "noise_network."),
    ("adapter_C", "adapters.C."),
    ("adapter_S1", "adapters.S1."),
    ("adapter_S2", "adapters.S2."),
    ("adapter_S3", "adapters.S3."),
)
NOISELORA_REQUIRED_PREFIXES = tuple(prefix for _, prefix in NOISELORA_REQUIRED_MODULES)
BASELINE_REQUIRED_MODULES = (("encoder", "encoder."),)
CHECKPOINT_CONFIG_SECTIONS = ("audio", "model", "loss", "optimizer", "scheduler", "training")


def _state_dict_from_checkpoint(obj):
    if isinstance(obj, dict):
        for key in ["model", "state_dict", "model_state_dict", "encoder", "speaker_encoder"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _clean_key(key):
    for prefix in ["module.", "model."]:
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def remap_legacy_ecapa_key(key):
    key = _clean_key(key)
    if key.startswith(IGNORED_LEGACY_PREFIXES):
        return None
    # Map legacy ECAPA checkpoints into the current module names.
    for old, new in LEGACY_ECAPA_PREFIXES:
        if key.startswith(old):
            key = new + key[len(old):]
            break
    return key.replace(".se.se.", ".se.net.")


def _candidate_keys(key, prefixes):
    key = remap_legacy_ecapa_key(key)
    if key is None:
        return
    # Try explicit remaps before plain prefix stripping.
    yield key
    for prefix in prefixes:
        if prefix and key.startswith(prefix):
            yield key[len(prefix):]


def _format_report(label, report):
    return (
        f"{label}: loaded={report['loaded_keys']} "
        f"params={report['loaded_params']}/{report['total_params']} "
        f"({report['loaded_percent']:.2f}%) "
        f"missing={len(report['missing_keys'])} unexpected={len(report['unexpected_keys'])}"
    )


def _countable_state_tensor(key, value):
    return hasattr(value, "numel") and value.numel() > 0 and not key.endswith("num_batches_tracked")


def _state_param_count(state):
    return sum(value.numel() for key, value in state.items() if _countable_state_tensor(key, value))


def _prefix_label(prefix):
    return prefix.rstrip(".").replace("adapters.", "adapter_").replace(".", "_")


def _module_specs(required_prefixes=None, required_modules=None):
    if required_modules is not None:
        return tuple(required_modules)
    return tuple((_prefix_label(prefix), prefix) for prefix in (required_prefixes or ()))


def _module_coverage(target, filtered, modules):
    reports = {}
    for name, prefix in modules:
        expected = {key: value for key, value in target.items() if key.startswith(prefix) and _countable_state_tensor(key, value)}
        loaded = {key: value for key, value in filtered.items() if key in expected and _countable_state_tensor(key, value)}
        expected_params = _state_param_count(expected)
        loaded_params = _state_param_count(loaded)
        coverage = 100.0 * loaded_params / max(expected_params, 1)
        reports[name] = {
            "prefix": prefix,
            "expected_tensors": len(expected),
            "loaded_tensors": len(loaded),
            "expected_params": int(expected_params),
            "loaded_params": int(loaded_params),
            "loaded_percent": float(coverage),
        }
    return reports


def _print_module_coverage(label, reports):
    for name, report in reports.items():
        print(
            f"{label}.{name}: {report['loaded_percent']:.2f}% "
            f"tensors={report['loaded_tensors']}/{report['expected_tensors']} "
            f"params={report['loaded_params']}/{report['expected_params']}"
        )


def _ensure_parent_dir(path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)


def _resume_required_modules(model):
    if hasattr(model, "noise_network") and hasattr(model, "adapters"):
        return NOISELORA_REQUIRED_MODULES
    if hasattr(model, "encoder"):
        return BASELINE_REQUIRED_MODULES
    return ()


def _require_resume_state(obj, label, scheduler_required):
    required = ["model", "criterion", "optimizer", "epoch"]
    if scheduler_required:
        # Require scheduler state only when a scheduler is configured.
        required.append("scheduler")
    missing = [key for key in required if key not in obj or obj[key] is None]
    if missing:
        raise RuntimeError(f"{label}: missing required training state: {', '.join(missing)}")
    if not scheduler_required and obj.get("scheduler") is not None:
        raise RuntimeError(f"{label}: checkpoint contains scheduler state but current scheduler is disabled")


def _load_criterion_state(criterion, state):
    if criterion is None:
        return
    if not isinstance(state, dict) or not state:
        raise RuntimeError("resume: criterion state is empty or invalid")
    try:
        # Restore the trainable AAM classifier strictly.
        criterion.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"resume: failed to restore criterion state strictly: {exc}") from exc


def sanitize_checkpoint_config(cfg):
    # Exclude local paths from checkpoint metadata.
    cfg = deepcopy(cfg or {})
    return {key: deepcopy(cfg[key]) for key in CHECKPOINT_CONFIG_SECTIONS if key in cfg}


def load_module_checkpoint(module, path, prefixes=None, strict=False, map_location="cpu", label="checkpoint"):
    if not path:
        raise ValueError(f"{label}: checkpoint path is empty")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label}: checkpoint not found: {path}")
    obj = torch.load(path, map_location=map_location)
    source = _state_dict_from_checkpoint(obj)
    if not isinstance(source, dict):
        raise TypeError(f"{label}: checkpoint does not contain a state dict")
    prefixes = tuple(prefixes or ())
    target = module.state_dict()
    filtered, used, ignored = {}, set(), set()
    for key, value in source.items():
        if _clean_key(key).startswith(IGNORED_LEGACY_PREFIXES):
            ignored.add(key)
            continue
        for candidate in _candidate_keys(key, prefixes):
            if candidate in target and tuple(target[candidate].shape) == tuple(value.shape):
                filtered[candidate] = value
                used.add(key)
                break
    if not filtered:
        raise RuntimeError(f"{label}: checkpoint loaded zero useful parameters from {path}")
    incompatible = module.load_state_dict(filtered, strict=strict)
    loaded_params = _state_param_count(filtered)
    total_params = _state_param_count(target)
    report = {
        "path": path,
        "loaded_keys": len(filtered),
        "loaded_params": int(loaded_params),
        "total_params": int(total_params),
        "loaded_percent": float(100.0 * loaded_params / max(total_params, 1)),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": [key for key in source.keys() if key not in used and key not in ignored],
        "ignored_keys": sorted(ignored),
    }
    print(_format_report(label, report))
    return report


def load_checkpoint(model, path, strict=False, map_location="cpu"):
    if not path:
        return [], []
    report = load_module_checkpoint(model, path, prefixes=("",), strict=strict, map_location=map_location)
    return report["missing_keys"], report["unexpected_keys"]


def load_full_model_checkpoint(
    model,
    path,
    required_prefixes=None,
    required_modules=None,
    min_loaded_percent=99.0,
    min_module_percent=99.0,
    strict=False,
    map_location="cpu",
    label="full_model",
):
    if not path:
        raise ValueError(f"{label}: checkpoint path is empty")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label}: checkpoint not found: {path}")
    obj = torch.load(path, map_location=map_location)
    source = _state_dict_from_checkpoint(obj)
    if not isinstance(source, dict):
        raise TypeError(f"{label}: checkpoint does not contain a state dict")
    target = model.state_dict()
    filtered, used = {}, set()
    for key, value in source.items():
        candidate = _clean_key(key)
        if candidate in target and tuple(target[candidate].shape) == tuple(value.shape):
            filtered[candidate] = value
            used.add(key)
    if not filtered:
        raise RuntimeError(f"{label}: checkpoint loaded zero useful parameters from {path}")
    loaded_params = _state_param_count(filtered)
    total_params = _state_param_count(target)
    loaded_percent = float(100.0 * loaded_params / max(total_params, 1))
    modules = _module_specs(required_prefixes, required_modules)
    module_reports = _module_coverage(target, filtered, modules)
    missing_required = [report["prefix"] for report in module_reports.values() if report["loaded_tensors"] == 0]
    incomplete = [
        f"{name} {report['loaded_percent']:.2f}%"
        for name, report in module_reports.items()
        if report["expected_params"] == 0 or report["loaded_percent"] < float(min_module_percent)
    ]
    unexpected = [key for key in source.keys() if key not in used]
    report = {
        "path": path,
        "loaded_keys": len(filtered),
        "loaded_params": int(loaded_params),
        "total_params": int(total_params),
        "loaded_percent": loaded_percent,
        "module_coverage": module_reports,
        "missing_required_prefixes": missing_required,
        "missing_keys": [key for key in target.keys() if key not in filtered],
        "unexpected_keys": unexpected,
    }
    print(
        f"{label}: loaded={report['loaded_keys']} "
        f"params={report['loaded_params']}/{report['total_params']} "
        f"({report['loaded_percent']:.2f}%) "
        f"missing={len(report['missing_keys'])} "
        f"unexpected={len(report['unexpected_keys'])}"
    )
    _print_module_coverage(label, module_reports)
    # Require high coverage for every inference-critical module.
    if incomplete:
        raise RuntimeError(f"{label}: incomplete required modules: {', '.join(incomplete)}")
    if loaded_percent < float(min_loaded_percent):
        raise RuntimeError(f"{label}: loaded parameter coverage {loaded_percent:.2f}% is below {min_loaded_percent:.2f}%")
    incompatible = model.load_state_dict(filtered, strict=strict)
    report["missing_keys"] = list(incompatible.missing_keys)
    return report


def load_complete_baseline_checkpoint(model, path, strict=False, map_location="cpu", label="baseline_eval"):
    # Baseline --checkpoint is a full training checkpoint, not ECAPA pretraining.
    return load_full_model_checkpoint(
        model,
        path,
        required_modules=BASELINE_REQUIRED_MODULES,
        min_loaded_percent=99.0,
        min_module_percent=99.0,
        strict=strict,
        map_location=map_location,
        label=label,
    )


def save_training_checkpoint(path, model, criterion, optimizer=None, scheduler=None, epoch=0, cfg=None, extra=None, scaler=None):
    _ensure_parent_dir(path)
    pack = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "criterion": criterion.state_dict() if criterion is not None else None,
        "config": sanitize_checkpoint_config(cfg),
    }
    if optimizer is not None:
        pack["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        pack["scheduler"] = scheduler.state_dict()
    if scaler is not None and scaler.is_enabled():
        pack["scaler"] = scaler.state_dict()
    if extra:
        extra = dict(extra)
        if "config" in extra:
            extra["config"] = sanitize_checkpoint_config(extra["config"])
        if "cfg" in extra:
            extra["cfg"] = sanitize_checkpoint_config(extra["cfg"])
        pack.update(extra)
    torch.save(pack, path)


def load_training_checkpoint(path, model, criterion=None, optimizer=None, scheduler=None, map_location="cpu", scaler=None):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    obj = torch.load(path, map_location=map_location)
    if not isinstance(obj, dict) or "model" not in obj:
        raise RuntimeError("Resume checkpoint must contain a full 'model' state dict")
    # Require complete training state for safe resume.
    _require_resume_state(obj, "resume", scheduler is not None)
    report = load_full_model_checkpoint(
        model,
        path,
        required_modules=_resume_required_modules(model),
        min_loaded_percent=99.0,
        min_module_percent=99.0,
        map_location=map_location,
        label="resume.model",
    )
    _load_criterion_state(criterion, obj["criterion"])
    if optimizer is not None:
        optimizer.load_state_dict(obj["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(obj["scheduler"])
    if scaler is not None and obj.get("scaler") is not None:
        scaler.load_state_dict(obj["scaler"])
    return {"epoch": int(obj.get("epoch", 0)), "model_report": report, "config": obj.get("config", {})}


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0, extra=None):
    _ensure_parent_dir(path)
    pack = {"epoch": int(epoch), "state_dict": model.state_dict()}
    if optimizer is not None:
        pack["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        pack["scheduler"] = scheduler.state_dict()
    if extra:
        extra = dict(extra)
        if "config" in extra:
            extra["config"] = sanitize_checkpoint_config(extra["config"])
        if "cfg" in extra:
            extra["cfg"] = sanitize_checkpoint_config(extra["cfg"])
        pack.update(extra)
    torch.save(pack, path)
