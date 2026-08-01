import torch

from losses import NoiseLoRASVLoss
from models import GlobalNoiseConditionedLoRA, HNCNoiseConditionedLoRA, build_model
from task import NoiseLoRASVTask


def tiny_cfg(use_noiselora=True):
    return {
        "audio": {
            "sample_rate": 16000,
            "n_fft": 512,
            "win_length_ms": 25,
            "hop_length_ms": 10,
            "n_mels": 80,
            "f_min": 20.0,
            "f_max": 7600.0,
            "preemphasis": 0.97,
            "log_eps": 1e-6,
            "mean_norm": True,
            "specaugment": {"enabled": False},
        },
        "model": {
            "use_noiselora": use_noiselora,
            "use_teacher": use_noiselora,
            "channels": 64,
            "embedding_dim": 192,
            "noise_dim": 16,
            "crn": {"base_channels": 4, "rnn_channels": 4, "rnn_hidden": 8},
            "lora": {"rank": 2, "alpha": 4, "hyper_hidden": 16, "gate_hidden": 8},
        },
        "loss": {
            "n_classes": 4,
            "aam_margin": 0.2,
            "aam_scale": 30,
            "infonce_temperature": 0.07,
            "speaker_weight": 1.0,
            "distill_weight": 1.0 if use_noiselora else 0.0,
            "noise_weight": 0.1 if use_noiselora else 0.0,
        },
    }


def test_noiselora_forward_backward_shapes_and_grads():
    cfg = tiny_cfg(True)
    model = build_model(cfg)
    model.train()
    noisy = torch.randn(2, 16000)
    clean = torch.randn(2, 16000)
    noise_target = noisy - clean
    labels = torch.tensor([0, 1])
    valid = torch.ones(2)
    outputs = model(noisy, clean=clean, aug=False)
    assert outputs["embedding"].shape == (2, 192)
    assert outputs["teacher_embedding"].shape == (2, 192)
    assert outputs["noise_hat"].shape[:2] == (2, 80)
    assert outputs["z_global"].shape == (2, 16)
    assert outputs["z_local"].shape[:2] == (2, 16)
    assert outputs["temporal_gates"]["S2"].shape[-1] > 0
    assert outputs["temporal_gates"]["S3"].shape[-1] > 0
    target_logmel = model.frontend(noise_target, aug=False)
    loss_pack = NoiseLoRASVLoss(cfg["loss"], embedding_dim=192)(outputs, labels, target_logmel, valid)
    assert torch.isfinite(loss_pack["loss"])
    loss_pack["loss"].backward()
    assert all(param.grad is None for param in model.teacher.parameters())
    assert any(param.grad is not None for param in model.student.parameters())
    assert any(param.grad is not None for param in model.noise_network.parameters())
    assert any(param.grad is not None for param in model.adapters.parameters())


def test_baseline_has_no_noiselora_modules():
    model = build_model(tiny_cfg(False))
    assert not any(isinstance(module, (GlobalNoiseConditionedLoRA, HNCNoiseConditionedLoRA)) for module in model.modules())


def test_eval_task_construction_does_not_require_teacher_checkpoint():
    cfg = tiny_cfg(True)
    cfg["paths"] = {"log_dir": ""}
    task = NoiseLoRASVTask(cfg, mode="eval")
    assert task.criterion is None
    assert task.optimizer is None
    assert not hasattr(task.model, "teacher") or task.model.teacher is None


def test_student_falls_back_to_teacher_checkpoint(tmp_path):
    cfg = tiny_cfg(True)
    cfg["loss"]["distill_weight"] = 0.0
    cfg["paths"] = {"teacher_checkpoint": str(tmp_path / "teacher.pth"), "log_dir": ""}
    source_model = build_model(tiny_cfg(False)).encoder
    torch.save({"state_dict": source_model.state_dict()}, cfg["paths"]["teacher_checkpoint"])
    task = NoiseLoRASVTask(cfg, mode="train")
    for key, value in source_model.state_dict().items():
        assert torch.equal(task.model.student.state_dict()[key], value)
