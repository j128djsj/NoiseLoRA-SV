def count_parameters(module, trainable_only=False):
    if module is None:
        return 0
    params = module.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def _adapter_count(model, names):
    if not hasattr(model, "adapters"):
        return 0
    return sum(count_parameters(model.adapters[name]) for name in names if name in model.adapters)


def _direct_parameter_count(module):
    if module is None:
        return 0
    return sum(param.numel() for param in module.parameters(recurse=False))


def _crn_excluding_msnrh(crn):
    if crn is None:
        return 0
    # Keep the CRN and MS-NRH counts non-overlapping.
    return sum(_direct_parameter_count(module) for name, module in crn.named_modules() if not name.startswith("msnrh"))


def summarize_model_parameters(model):
    student = getattr(model, "student", getattr(model, "encoder", None))
    teacher = getattr(model, "teacher", None)
    crn = getattr(model, "noise_network", None)
    msnrh = getattr(crn, "msnrh", None)
    total_training = count_parameters(model)
    total_inference = total_training - count_parameters(teacher)
    return {
        "ecapa_student": count_parameters(student),
        "frozen_teacher": count_parameters(teacher),
        "crn_excluding_ms_nrh": _crn_excluding_msnrh(crn),
        "ms_nrh": count_parameters(msnrh),
        "global_lora_blocks": _adapter_count(model, ["C", "S1"]),
        "hnc_lora_blocks": _adapter_count(model, ["S2", "S3"]),
        "total_training_model": total_training,
        "trainable_training_model": count_parameters(model, trainable_only=True),
        "total_inference_model_excluding_teacher": total_inference,
    }


def format_parameter_summary(summary):
    lines = []
    for key, value in summary.items():
        lines.append(f"{key}: {value:,} ({value / 1_000_000:.2f}M)")
    lines.append("paper_ecapa_reference: 14,730,000 (14.73M)")
    lines.append("paper_noiselora_ecapa_reference: 24,190,000 (24.19M)")
    return "\n".join(lines)
