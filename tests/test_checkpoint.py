import pytest
import torch
import torch.nn as nn

from models import ECAPATDNN
from losses import NoiseLoRASVLoss
from utils.checkpoint import NOISELORA_REQUIRED_PREFIXES, load_complete_baseline_checkpoint, load_full_model_checkpoint
from utils.checkpoint import load_module_checkpoint, save_checkpoint
from utils.checkpoint import load_training_checkpoint, save_training_checkpoint


def test_checkpoint_load_reports_useful_keys(tmp_path):
    module = nn.Linear(2, 2)
    path = tmp_path / "linear.pth"
    torch.save({"model": {"student.weight": torch.ones_like(module.weight), "student.bias": torch.zeros_like(module.bias)}}, path)
    report = load_module_checkpoint(module, path, prefixes=("student.",), label="linear")
    assert report["loaded_keys"] == 2


def test_checkpoint_zero_useful_keys_raises(tmp_path):
    module = nn.Linear(2, 2)
    path = tmp_path / "bad.pth"
    torch.save({"model": {"other.weight": torch.ones(2, 2)}}, path)
    with pytest.raises(RuntimeError, match="zero useful"):
        load_module_checkpoint(module, path, prefixes=("student.",), label="bad")


def _legacy_key(key):
    pairs = [
        ("C.0.", "speaker_encoder.conv1."),
        ("C.2.", "speaker_encoder.bn1."),
        ("S1.", "speaker_encoder.layer1."),
        ("S2.", "speaker_encoder.layer2."),
        ("S3.", "speaker_encoder.layer3."),
        ("mfa.0.", "speaker_encoder.layer4."),
        ("pool.attention.", "speaker_encoder.attention."),
        ("bn.", "speaker_encoder.bn5."),
        ("linear.", "speaker_encoder.fc6."),
        ("out_bn.", "speaker_encoder.bn6."),
    ]
    for new, old in pairs:
        if key.startswith(new):
            return (old + key[len(new):]).replace(".se.net.", ".se.se.")
    return key


def test_legacy_ecapa_checkpoint_remaps_all_compatible_keys(tmp_path):
    model = ECAPATDNN(channels=64, embedding_dim=192)
    legacy = {}
    expected = {}
    for key, value in model.state_dict().items():
        mapped = _legacy_key(key)
        tensor = torch.ones_like(value)
        legacy[mapped] = tensor
        expected[key] = tensor
    legacy["speaker_encoder.torchfbank.0.kernel"] = torch.ones(1)
    path = tmp_path / "legacy_ecapa.pth"
    torch.save({"state_dict": legacy}, path)
    report = load_module_checkpoint(model, path, prefixes=("speaker_encoder.",), label="legacy_ecapa")
    assert report["loaded_keys"] == len(expected)
    assert report["loaded_percent"] == pytest.approx(100.0)
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_resume_restores_model_and_criterion(tmp_path):
    model = nn.Linear(2, 2)
    criterion = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    path = tmp_path / "resume.pth"
    save_training_checkpoint(path, model, criterion, optimizer=optimizer, scheduler=scheduler, epoch=7, cfg={"ok": True})
    new_model = nn.Linear(2, 2)
    new_criterion = nn.Linear(2, 1)
    new_optimizer = torch.optim.Adam(list(new_model.parameters()) + list(new_criterion.parameters()), lr=0.01)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1)
    state = load_training_checkpoint(path, new_model, new_criterion, new_optimizer, new_scheduler)
    assert state["epoch"] == 7
    for old, new in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(old, new)
    for old, new in zip(criterion.parameters(), new_criterion.parameters()):
        assert torch.allclose(old, new)


class TinyNoiseLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = nn.Linear(128, 128)
        self.noise_network = nn.Linear(2, 2)
        self.adapters = nn.ModuleDict({
            "C": nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)),
            "S1": nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)),
            "S2": nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)),
            "S3": nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)),
        })


def test_complete_noiselora_checkpoint_passes_validation(tmp_path):
    model = TinyNoiseLoRA()
    path = tmp_path / "full.pth"
    torch.save({"model": model.state_dict()}, path)
    report = load_full_model_checkpoint(model, path, required_prefixes=NOISELORA_REQUIRED_PREFIXES)
    assert report["missing_required_prefixes"] == []
    assert report["loaded_percent"] == pytest.approx(100.0)
    assert report["module_coverage"]["adapter_S2"]["loaded_percent"] == pytest.approx(100.0)


def test_incomplete_noiselora_checkpoint_is_rejected(tmp_path):
    model = TinyNoiseLoRA()
    path = tmp_path / "student_only.pth"
    partial = {key: value for key, value in model.state_dict().items() if key.startswith("student.")}
    torch.save({"model": partial}, path)
    with pytest.raises(RuntimeError, match="incomplete required modules"):
        load_full_model_checkpoint(model, path, required_prefixes=NOISELORA_REQUIRED_PREFIXES)


def test_missing_adapter_block_is_rejected(tmp_path):
    model = TinyNoiseLoRA()
    path = tmp_path / "missing_adapter.pth"
    partial = {key: value for key, value in model.state_dict().items() if not key.startswith("adapters.S2.")}
    torch.save({"model": partial}, path)
    with pytest.raises(RuntimeError, match="adapter_S2 0.00%"):
        load_full_model_checkpoint(model, path, required_prefixes=NOISELORA_REQUIRED_PREFIXES)


def test_adapter_with_only_num_batches_tracked_is_rejected(tmp_path):
    model = TinyNoiseLoRA()
    path = tmp_path / "adapter_buffer_only.pth"
    partial = {key: value for key, value in model.state_dict().items() if not key.startswith("adapters.S2.")}
    partial["adapters.S2.1.num_batches_tracked"] = model.state_dict()["adapters.S2.1.num_batches_tracked"]
    torch.save({"model": partial}, path)
    with pytest.raises(RuntimeError, match="adapter_S2 0.00%"):
        load_full_model_checkpoint(model, path, required_prefixes=NOISELORA_REQUIRED_PREFIXES)


def test_incomplete_noise_network_is_rejected(tmp_path):
    model = TinyNoiseLoRA()
    path = tmp_path / "incomplete_noise.pth"
    partial = {key: value for key, value in model.state_dict().items() if key != "noise_network.weight"}
    torch.save({"model": partial}, path)
    with pytest.raises(RuntimeError, match="noise_network"):
        load_full_model_checkpoint(model, path, required_prefixes=NOISELORA_REQUIRED_PREFIXES)


def test_high_overall_coverage_still_rejects_incomplete_adapter(tmp_path):
    model = TinyNoiseLoRA()
    path = tmp_path / "high_overall_missing_adapter.pth"
    partial = {key: value for key, value in model.state_dict().items() if not key.startswith("adapters.S3.")}
    torch.save({"model": partial}, path)
    with pytest.raises(RuntimeError, match="adapter_S3"):
        load_full_model_checkpoint(model, path, required_prefixes=NOISELORA_REQUIRED_PREFIXES, min_loaded_percent=80.0)


class TinyBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = nn.BatchNorm1d(2)
        self.encoder = nn.Linear(2, 2)


def _train_parts(model, scheduler_enabled=True, criterion=None):
    criterion = criterion or nn.Linear(2, 1)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97) if scheduler_enabled else None
    return criterion, optimizer, scheduler


def _save_resume(path, model, epoch=3, scheduler_enabled=True, criterion=None):
    criterion, optimizer, scheduler = _train_parts(model, scheduler_enabled=scheduler_enabled, criterion=criterion)
    save_training_checkpoint(path, model, criterion, optimizer=optimizer, scheduler=scheduler, epoch=epoch)
    return criterion, optimizer, scheduler


def _load_resume(path, model, scheduler_enabled=True, criterion=None):
    criterion, optimizer, scheduler = _train_parts(model, scheduler_enabled=scheduler_enabled, criterion=criterion)
    return load_training_checkpoint(path, model, criterion, optimizer, scheduler)


def _aam_loss(n_classes=3, embedding_dim=2):
    return NoiseLoRASVLoss(
        {"n_classes": n_classes, "distill_weight": 0.0, "noise_weight": 0.0},
        embedding_dim=embedding_dim,
    )


def test_complete_noiselora_resume_checkpoint_passes(tmp_path):
    path = tmp_path / "resume_noiselora.pth"
    _save_resume(path, TinyNoiseLoRA())
    report = _load_resume(path, TinyNoiseLoRA())["model_report"]
    assert report["module_coverage"]["adapter_C"]["loaded_percent"] == pytest.approx(100.0)
    assert report["module_coverage"]["noise_network"]["loaded_percent"] == pytest.approx(100.0)


def test_complete_aam_criterion_state_passes(tmp_path):
    path = tmp_path / "resume_aam.pth"
    source_loss = _aam_loss()
    _save_resume(path, TinyBaseline(), criterion=source_loss)
    target_loss = _aam_loss()
    _load_resume(path, TinyBaseline(), criterion=target_loss)
    assert torch.allclose(source_loss.state_dict()["speaker.weight"], target_loss.state_dict()["speaker.weight"])


def test_empty_criterion_state_is_rejected(tmp_path):
    path = tmp_path / "resume_empty_criterion.pth"
    _save_resume(path, TinyBaseline(), criterion=_aam_loss())
    pack = torch.load(path, map_location="cpu")
    pack["criterion"] = {}
    torch.save(pack, path)
    with pytest.raises(RuntimeError, match="criterion state is empty"):
        _load_resume(path, TinyBaseline(), criterion=_aam_loss())


def test_missing_aam_classifier_weight_is_rejected(tmp_path):
    path = tmp_path / "resume_missing_aam_weight.pth"
    _save_resume(path, TinyBaseline(), criterion=_aam_loss())
    pack = torch.load(path, map_location="cpu")
    pack["criterion"].pop("speaker.weight")
    torch.save(pack, path)
    with pytest.raises(RuntimeError, match="criterion state"):
        _load_resume(path, TinyBaseline(), criterion=_aam_loss())


def test_incompatible_aam_classifier_shape_is_rejected(tmp_path):
    path = tmp_path / "resume_bad_aam_shape.pth"
    _save_resume(path, TinyBaseline(), criterion=_aam_loss())
    pack = torch.load(path, map_location="cpu")
    pack["criterion"]["speaker.weight"] = torch.randn(4, 3)
    torch.save(pack, path)
    with pytest.raises(RuntimeError, match="criterion state"):
        _load_resume(path, TinyBaseline(), criterion=_aam_loss())


def test_noiselora_resume_missing_adapter_is_rejected(tmp_path):
    path = tmp_path / "resume_missing_adapter.pth"
    _save_resume(path, TinyNoiseLoRA())
    pack = torch.load(path, map_location="cpu")
    pack["model"] = {key: value for key, value in pack["model"].items() if not key.startswith("adapters.S2.")}
    torch.save(pack, path)
    with pytest.raises(RuntimeError, match="adapter_S2"):
        _load_resume(path, TinyNoiseLoRA())


def test_noiselora_resume_incomplete_noise_network_is_rejected(tmp_path):
    path = tmp_path / "resume_incomplete_noise.pth"
    _save_resume(path, TinyNoiseLoRA())
    pack = torch.load(path, map_location="cpu")
    pack["model"].pop("noise_network.weight")
    torch.save(pack, path)
    with pytest.raises(RuntimeError, match="noise_network"):
        _load_resume(path, TinyNoiseLoRA())


def test_noiselora_resume_student_only_is_rejected(tmp_path):
    path = tmp_path / "resume_student_only.pth"
    _save_resume(path, TinyNoiseLoRA())
    pack = torch.load(path, map_location="cpu")
    pack["model"] = {key: value for key, value in pack["model"].items() if key.startswith("student.")}
    torch.save(pack, path)
    with pytest.raises(RuntimeError, match="noise_network"):
        _load_resume(path, TinyNoiseLoRA())


def test_resume_missing_optimizer_state_is_rejected(tmp_path):
    path = tmp_path / "resume_missing_optimizer.pth"
    _save_resume(path, TinyNoiseLoRA())
    pack = torch.load(path, map_location="cpu")
    pack.pop("optimizer")
    torch.save(pack, path)
    with pytest.raises(RuntimeError, match="optimizer"):
        _load_resume(path, TinyNoiseLoRA())


def test_resume_missing_criterion_state_is_rejected(tmp_path):
    path = tmp_path / "resume_missing_criterion.pth"
    _save_resume(path, TinyNoiseLoRA())
    pack = torch.load(path, map_location="cpu")
    pack.pop("criterion")
    torch.save(pack, path)
    with pytest.raises(RuntimeError, match="criterion"):
        _load_resume(path, TinyNoiseLoRA())


def test_resume_missing_exponential_scheduler_state_is_rejected(tmp_path):
    path = tmp_path / "resume_missing_scheduler.pth"
    _save_resume(path, TinyNoiseLoRA())
    pack = torch.load(path, map_location="cpu")
    pack.pop("scheduler")
    torch.save(pack, path)
    with pytest.raises(RuntimeError, match="scheduler"):
        _load_resume(path, TinyNoiseLoRA())


def test_resume_without_scheduler_passes_when_disabled(tmp_path):
    path = tmp_path / "resume_no_scheduler.pth"
    _save_resume(path, TinyNoiseLoRA(), scheduler_enabled=False)
    report = _load_resume(path, TinyNoiseLoRA(), scheduler_enabled=False)["model_report"]
    assert report["module_coverage"]["adapter_S1"]["loaded_percent"] == pytest.approx(100.0)


def test_complete_baseline_resume_checkpoint_passes(tmp_path):
    path = tmp_path / "resume_baseline.pth"
    _save_resume(path, TinyBaseline())
    report = _load_resume(path, TinyBaseline())["model_report"]
    assert report["module_coverage"]["encoder"]["loaded_percent"] == pytest.approx(100.0)


def test_exponential_scheduler_state_restores(tmp_path):
    model = TinyBaseline()
    criterion, optimizer, scheduler = _train_parts(model)
    optimizer.zero_grad()
    scheduler.step()
    path = tmp_path / "resume_exponential_scheduler.pth"
    save_training_checkpoint(path, model, criterion, optimizer=optimizer, scheduler=scheduler, epoch=2)
    new_model = TinyBaseline()
    new_criterion, new_optimizer, new_scheduler = _train_parts(new_model)
    load_training_checkpoint(path, new_model, new_criterion, new_optimizer, new_scheduler)
    assert new_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


def test_encoder_only_baseline_checkpoint_is_rejected_for_resume(tmp_path):
    path = tmp_path / "resume_encoder_only.pth"
    _save_resume(path, TinyBaseline())
    pack = torch.load(path, map_location="cpu")
    pack["model"] = {key: value for key, value in pack["model"].items() if key.startswith("encoder.")}
    torch.save(pack, path)
    with pytest.raises(RuntimeError, match="coverage"):
        _load_resume(path, TinyBaseline())


def test_complete_baseline_checkpoint_passes_validation(tmp_path):
    model = TinyBaseline()
    path = tmp_path / "baseline_full.pth"
    torch.save({"model": model.state_dict()}, path)
    report = load_complete_baseline_checkpoint(model, path)
    assert report["module_coverage"]["encoder"]["loaded_percent"] == pytest.approx(100.0)


def test_encoder_only_pretraining_loads_into_baseline_encoder(tmp_path):
    model = TinyBaseline()
    path = tmp_path / "encoder_pretrain.pth"
    torch.save({"state_dict": {"encoder.weight": torch.ones_like(model.encoder.weight)}}, path)
    report = load_module_checkpoint(model.encoder, path, prefixes=("encoder.",), label="encoder_pretrain")
    assert report["loaded_keys"] == 1


def test_one_key_baseline_checkpoint_is_rejected(tmp_path):
    model = TinyBaseline()
    path = tmp_path / "baseline_one_key.pth"
    torch.save({"model": {"encoder.weight": model.state_dict()["encoder.weight"]}}, path)
    with pytest.raises(RuntimeError, match="encoder"):
        load_complete_baseline_checkpoint(model, path)


def test_frontend_only_baseline_checkpoint_is_rejected(tmp_path):
    model = TinyBaseline()
    path = tmp_path / "baseline_frontend_only.pth"
    torch.save({"model": {"frontend.running_mean": model.state_dict()["frontend.running_mean"]}}, path)
    with pytest.raises(RuntimeError, match="encoder"):
        load_complete_baseline_checkpoint(model, path)


def test_checkpoint_metadata_sanitizes_private_paths(tmp_path):
    model = nn.Linear(2, 2)
    criterion = nn.Linear(2, 1)
    windows_root = "D:" + "\\private"
    public_home = "/public" + "/ho" + "me/user/test"
    home_ckpt = "/ho" + "me/user/teacher.pth"
    cfg = {
        "_config_dir": windows_root + "\\repo",
        "paths": {
            "train_root": windows_root + "\\data",
            "test_root": public_home,
            "teacher_checkpoint": home_ckpt,
        },
        "model": {"embedding_dim": 192},
        "training": {"epochs": 1},
        "audio": {"n_mels": 80},
    }
    original = cfg["paths"]["train_root"]
    path = tmp_path / "private_meta.pth"
    save_training_checkpoint(path, model, criterion, cfg=cfg)
    pack = torch.load(path, map_location="cpu")
    assert cfg["paths"]["train_root"] == original
    assert "paths" not in pack["config"]
    assert "_config_dir" not in pack["config"]
    text = repr(pack["config"])
    assert "D:" + "\\" not in text
    assert "/ho" + "me/" not in text
    assert "/public" + "/ho" + "me/" not in text
    assert pack["config"]["model"]["embedding_dim"] == 192
    assert pack["config"]["training"]["epochs"] == 1


def test_checkpoint_saves_without_parent_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = nn.Linear(2, 2)
    save_checkpoint("model.pth", model)
    assert (tmp_path / "model.pth").is_file()
    assert "state_dict" in torch.load("model.pth", map_location="cpu")
    (tmp_path / "model.pth").unlink()


def test_checkpoint_saves_in_nested_directory(tmp_path):
    model = nn.Linear(2, 2)
    path = tmp_path / "checkpoints" / "model.pth"
    save_checkpoint(path, model)
    assert path.is_file()
    assert "state_dict" in torch.load(path, map_location="cpu")


def test_training_checkpoint_saves_without_parent_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = nn.Linear(2, 2)
    criterion, optimizer, scheduler = _train_parts(model)
    save_training_checkpoint("train.pth", model, criterion, optimizer=optimizer, scheduler=scheduler, epoch=1)
    pack = torch.load("train.pth", map_location="cpu")
    assert pack["epoch"] == 1
    assert "optimizer" in pack and "scheduler" in pack and "criterion" in pack
    (tmp_path / "train.pth").unlink()
