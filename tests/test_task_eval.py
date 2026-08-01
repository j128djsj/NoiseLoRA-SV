import pytest
import torch

import task as task_module
from DatasetLoader import build_eval_loader
from task import NoiseLoRASVTask


class DummyLogger:
    def info(self, *args, **kwargs):
        pass


class DummyModel:
    def eval(self):
        pass

    def extract_embedding(self, wav):
        return torch.ones(wav.size(0), 192)


class DummyDataset:
    def __init__(self, trials):
        self.trials = trials


def tiny_cfg():
    return {
        "audio": {"n_mels": 80, "specaugment": {"enabled": False}},
        "model": {"use_noiselora": False, "channels": 64, "embedding_dim": 192},
        "loss": {"n_classes": 4, "distill_weight": 0.0, "noise_weight": 0.0},
    }


def _fake_task(trials, monkeypatch):
    obj = NoiseLoRASVTask.__new__(NoiseLoRASVTask)
    obj.cfg = tiny_cfg()
    obj.device = torch.device("cpu")
    obj.model = DummyModel()
    obj.logger = DummyLogger()
    obj._eval_checkpoint_loaded = True
    obj.pretrained_eval = False
    loader = [(torch.zeros(1, 160), ["a"])]
    monkeypatch.setattr(task_module, "build_eval_loader", lambda *args, **kwargs: (loader, DummyDataset(trials)))
    return obj


def test_evaluation_raises_when_no_trials_scored(monkeypatch):
    obj = _fake_task([(1, "missing", "a")], monkeypatch)
    with pytest.raises(RuntimeError, match="No verification trials"):
        obj.evaluate(condition="clean")


def test_evaluation_requires_positive_and_negative_trials(monkeypatch):
    obj = _fake_task([(1, "a", "a")], monkeypatch)
    with pytest.raises(RuntimeError, match="positive and negative"):
        obj.evaluate(condition="clean")


def test_eval_requires_checkpoint_unless_pretrained_mode(monkeypatch):
    obj = _fake_task([(1, "a", "a")], monkeypatch)
    obj._eval_checkpoint_loaded = False
    with pytest.raises(RuntimeError, match="requires --checkpoint"):
        obj.evaluate(condition="clean")


def test_evaluation_batch_size_must_be_one():
    cfg = {"evaluation": {"batch_size": 2}, "audio": {}, "paths": {}}
    with pytest.raises(ValueError, match="batch_size must be 1"):
        build_eval_loader(cfg)
