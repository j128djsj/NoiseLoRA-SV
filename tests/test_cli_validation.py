from argparse import Namespace

import pytest

from main import validate_args


def _args(**kwargs):
    data = {
        "mode": "eval",
        "condition": None,
        "noise_type": None,
        "snr": None,
        "checkpoint": "",
        "resume": "",
        "pretrained_eval": False,
    }
    data.update(kwargs)
    return Namespace(**data)


def _cfg(use_noiselora=False):
    return {"model": {"use_noiselora": use_noiselora}, "evaluation": {"snrs": [0, 5, 10, 15, 20]}}


def test_cli_rejects_checkpoint_in_train():
    with pytest.raises(ValueError, match="checkpoint"):
        validate_args(_args(mode="train", checkpoint="x.pth"), _cfg())


def test_cli_allows_plain_train_and_resume():
    validate_args(_args(mode="train"), _cfg())
    validate_args(_args(mode="train", resume="last.pth"), _cfg())


def test_cli_rejects_eval_only_args_in_train():
    with pytest.raises(ValueError, match="snr"):
        validate_args(_args(mode="train", snr=10), _cfg())
    with pytest.raises(ValueError, match="noise-type"):
        validate_args(_args(mode="train", noise_type="music"), _cfg())
    with pytest.raises(ValueError, match="pretrained-eval"):
        validate_args(_args(mode="train", pretrained_eval=True), _cfg())
    with pytest.raises(ValueError, match="condition"):
        validate_args(_args(mode="train", condition="clean"), _cfg())


def test_cli_rejects_resume_in_eval():
    with pytest.raises(ValueError, match="resume"):
        validate_args(_args(resume="x.pth"), _cfg())


def test_cli_rejects_pretrained_eval_for_noiselora():
    with pytest.raises(ValueError, match="baseline ECAPA"):
        validate_args(_args(pretrained_eval=True), _cfg(use_noiselora=True))


def test_cli_rejects_clean_noise_options():
    with pytest.raises(ValueError, match="snr"):
        validate_args(_args(condition="clean", snr=10), _cfg())
    with pytest.raises(ValueError, match="noise-type"):
        validate_args(_args(condition="clean", noise_type="music"), _cfg())


def test_cli_rejects_unseen_noise_type_and_bad_snr():
    with pytest.raises(ValueError, match="noise-type"):
        validate_args(_args(condition="unseen", noise_type="noise", snr=10), _cfg())
    with pytest.raises(ValueError, match="evaluation.snrs"):
        validate_args(_args(condition="seen", noise_type="noise", snr=7), _cfg())


def test_cli_requires_noiselora_eval_checkpoint():
    with pytest.raises(ValueError, match="requires --checkpoint"):
        validate_args(_args(condition="seen", noise_type="noise", snr=10), _cfg(use_noiselora=True))


def test_cli_allows_normal_eval_modes():
    validate_args(_args(condition="clean", checkpoint="full.pth"), _cfg())
    validate_args(_args(condition="seen", noise_type="noise", snr=10, checkpoint="full.pth"), _cfg())
    validate_args(_args(condition="unseen", snr=10, checkpoint="full.pth"), _cfg())
