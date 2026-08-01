import pytest

from utils.config import validate_config


def _cfg(**kwargs):
    cfg = {
        "audio": {"n_mels": 80},
        "model": {"use_noiselora": False},
        "optimizer": {"speaker_lr_scale": 1.0},
        "paths": {},
    }
    cfg.update(kwargs)
    return cfg


def test_config_accepts_paper_mel_dimension():
    validate_config(_cfg())


def test_config_rejects_non_paper_mel_dimension():
    with pytest.raises(ValueError, match="audio.n_mels=80"):
        validate_config(_cfg(audio={"n_mels": 64}))


def test_random_baseline_reduced_speaker_lr_warns():
    cfg = _cfg(optimizer={"speaker_lr_scale": 0.1})
    with pytest.warns(RuntimeWarning, match="speaker_lr_scale=1.0"):
        validate_config(cfg)


def test_pretrained_baseline_reduced_speaker_lr_is_allowed():
    cfg = _cfg(optimizer={"speaker_lr_scale": 0.1}, paths={"student_checkpoint": "pretrained.pth"})
    validate_config(cfg)
