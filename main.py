import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser(description="NoiseLoRA-SV")
    parser.add_argument("--config", default="configs/noiselora_ecapa.yaml")
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--condition", choices=["clean", "seen", "unseen"], default=None)
    parser.add_argument("--noise-type", choices=["babble", "music", "noise"], default=None)
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--pretrained-eval", action="store_true")
    return parser.parse_args()


def _snr_allowed(snr, configured):
    return any(float(snr) == float(value) for value in configured)


def validate_args(args, cfg):
    use_noiselora = bool(cfg.get("model", {}).get("use_noiselora", False))
    configured_snrs = cfg.get("evaluation", {}).get("snrs", [0, 5, 10, 15, 20])
    if args.mode == "train":
        # Keep evaluation-only flags out of training runs.
        if args.checkpoint:
            raise ValueError("--checkpoint is only valid in evaluation mode; use --resume for training.")
        if args.pretrained_eval:
            raise ValueError("--pretrained-eval is only valid in evaluation mode.")
        if args.noise_type is not None:
            raise ValueError("--noise-type is only valid in evaluation mode.")
        if args.snr is not None:
            raise ValueError("--snr is only valid in evaluation mode.")
        if args.condition is not None:
            raise ValueError("--condition is only valid in evaluation mode.")
        return
    args.condition = args.condition or "clean"
    if args.mode == "eval" and args.resume:
        raise ValueError("--resume is only for training")
    if args.pretrained_eval and args.mode != "eval":
        raise ValueError("--pretrained-eval is only for evaluation")
    if args.pretrained_eval and use_noiselora:
        raise ValueError("--pretrained-eval is baseline ECAPA only")
    if args.condition == "clean" and args.snr is not None:
        raise ValueError("--snr is not used for clean evaluation")
    if args.condition == "clean" and args.noise_type is not None:
        raise ValueError("--noise-type is not used for clean evaluation")
    if args.condition == "unseen" and args.noise_type is not None:
        raise ValueError("--noise-type is not used for unseen evaluation")
    if args.snr is not None and not _snr_allowed(args.snr, configured_snrs):
        raise ValueError("--snr must be one of evaluation.snrs")
    # NoiseLoRA eval needs the full trained graph, not speaker pretraining weights.
    if args.mode == "eval" and use_noiselora and not args.checkpoint:
        raise ValueError("NoiseLoRA evaluation requires --checkpoint")


def main():
    args = parse_args()
    from noiselora_sv.training.task import NoiseLoRASVTask
    from noiselora_sv.utils.config import load_config

    cfg = load_config(args.config)
    validate_args(args, cfg)
    task = NoiseLoRASVTask(cfg, mode=args.mode, resume=args.resume, pretrained_eval=args.pretrained_eval)
    if args.mode == "train":
        task.train()
        return
    eval_cfg = cfg.get("evaluation", {})
    if args.condition == "clean":
        results = [task.evaluate(condition="clean", checkpoint=args.checkpoint)]
    else:
        snrs = [args.snr] if args.snr is not None else eval_cfg.get("snrs", [0, 5, 10, 15, 20])
        noise_types = [args.noise_type] if args.noise_type else eval_cfg.get("seen_noise_types", ["babble", "music", "noise"])
        if args.condition == "unseen":
            noise_types = [None]
        results = [
            task.evaluate(
                condition=args.condition,
                snr=snr,
                checkpoint=args.checkpoint if idx == 0 else "",
                noise_type=noise_type,
            )
            for idx, (snr, noise_type) in enumerate((s, n) for n in noise_types for s in snrs)
        ]
    for result in results:
        print(
            f"EER={result['eer']:.4f}% condition={result['condition']} "
            f"noise_type={result['noise_type']} snr={result['snr']} "
            f"trials={result['trials']} pos={result['positive_trials']} "
            f"neg={result['negative_trials']} skipped={result['skipped_trials']}"
        )


if __name__ == "__main__":
    main()
