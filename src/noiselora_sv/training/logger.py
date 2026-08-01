import json
import os
import time


class JsonlLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir or "outputs/logs"
        os.makedirs(self.log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(self.log_dir, f"run-{stamp}.jsonl")

    def log(self, **fields):
        fields.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(fields, ensure_ascii=True) + "\n")

    def info(self, message, **fields):
        self.log(level="info", message=message, **fields)


def build_logger(cfg=None):
    cfg = cfg or {}
    return JsonlLogger(cfg.get("log_dir", "outputs/logs"))
