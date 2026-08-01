from types import SimpleNamespace

import task as task_module
from task import NoiseLoRASVTask


class DummyLogger:
    def __init__(self):
        self.calls = []

    def info(self, name, **kwargs):
        self.calls.append((name, kwargs))


class DummySampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


def test_one_epoch_training_flow_logs_and_saves(tmp_path, monkeypatch):
    obj = NoiseLoRASVTask.__new__(NoiseLoRASVTask)
    obj.mode = "train"
    obj.cfg = {"training": {"epochs": 1}, "paths": {"checkpoint_dir": str(tmp_path)}}
    obj.start_epoch = 0
    obj.scheduler = None
    obj.logger = DummyLogger()
    obj.model = object()
    obj.criterion = object()
    obj.optimizer = object()
    obj.scaler = None
    loader = SimpleNamespace(batch_sampler=DummySampler())
    saved = []

    monkeypatch.setattr(task_module, "build_train_loader", lambda cfg: loader)
    monkeypatch.setattr(NoiseLoRASVTask, "train_epoch", lambda self, loader, epoch: {"loss": 1.0, "top1": 50.0})
    monkeypatch.setattr(task_module, "save_training_checkpoint", lambda *args, **kwargs: saved.append((args, kwargs)))

    obj.train()

    assert obj.logger.calls == [("epoch", {"epoch": 1, "loss": 1.0, "top1": 50.0})]
    assert loader.batch_sampler.epochs == [1]
    assert saved and saved[0][0][0].endswith("last.pth")
