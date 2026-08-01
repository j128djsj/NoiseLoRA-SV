# Reproduction Guide

This document describes how to reproduce the NoiseLoRA-SV training and evaluation pipeline using the official source code.

## Dataset Preparation

Configure dataset and trial paths in YAML before running training or evaluation:

- `paths.train_root` or `paths.train_list`
- `paths.test_root`
- `paths.trials`
- `paths.musan_train_root`
- `paths.musan_test_root`
- `paths.unseen_noise_root`
- `paths.teacher_checkpoint` when using a trained clean teacher

Datasets and pretrained checkpoints are not bundled in this repository. Keep local dataset, checkpoint, and output paths in YAML rather than hard-coding them in runtime code.

## MUSAN And NonSpeech100

Seen-noise evaluation uses MUSAN and can be run separately for Babble, Music, and Noise. `musan_train_root` and `musan_test_root` must be explicit, separate MUSAN split directories; the runtime does not fall back to one shared full MUSAN directory.

Unseen-noise evaluation uses `paths.unseen_noise_root`, for example a prepared NonSpeech100-style noise directory. Use the configured `evaluation.snrs` values or pass `--snr` explicitly for a single SNR condition.

## Trial Lists

Evaluation reads strict trial files with exactly three fields per line:

```text
<label> <relative_audio_1> <relative_audio_2>
```

`label` must be `1` for same-speaker trials or `0` for different-speaker trials. Missing audio files are treated as configuration errors.

## EER Evaluation

The evaluator extracts one embedding per utterance, computes cosine similarity for each trial, and reports equal error rate (EER). Scores must be finite; NaN and infinite scores are rejected before threshold sorting. Scores are sorted by threshold, samples with the same score are updated as one group, and operating points above the maximum score and below the minimum score are included. If FAR and FRR cross between adjacent operating points, linear interpolation is used.

Clean trials use the original waveform. Seen noisy trials can be evaluated separately as `babble`, `music`, or `noise`; unseen noisy trials use `paths.unseen_noise_root`.

## Training Configuration

`configs/baseline_ecapa.yaml` keeps `training.augment_prob: 0.0` so the baseline is a clean ECAPA setting unless augmentation is explicitly enabled. `configs/noiselora_ecapa.yaml` follows the paper by setting `training.augment_prob: 1.0`, dynamically mixing every 3-second crop with MUSAN training noise at a uniformly sampled 0-20 dB SNR, and using the paper model frontend with `audio.n_mels=80`.

The clean ECAPA baseline uses `optimizer.speaker_lr_scale: 1.0` for from-scratch training. The NoiseLoRA-SV configuration keeps `speaker_lr_scale: 0.1` for lightly fine-tuning a pretrained speaker encoder while training the noise-conditioned modules.

## Checkpoints

NoiseLoRA-SV evaluation requires a full trained checkpoint containing useful parameters for the ECAPA student, CRN noise network, and all four adapter insertion points. Each inference-critical module is validated separately with high tensor and parameter coverage, so a checkpoint cannot pass merely by loading one small buffer. ECAPA-only speaker-pretraining checkpoints are accepted for baseline ECAPA evaluation with `--pretrained-eval`, but they are rejected for NoiseLoRA-SV evaluation.

Baseline `--checkpoint` evaluation expects a complete trained baseline checkpoint with near-complete model coverage. Clean ECAPA pretraining weights remain a separate use case and are loaded with legacy key remapping only through initialization or `--pretrained-eval`.

The implementation prints loaded tensor counts, loaded parameter percentage, per-module coverage, missing keys, and unexpected keys during checkpoint loading. A checkpoint that loads zero useful parameters is always rejected.

Resume checkpoints require complete training state: model, criterion, optimizer, scheduler, and epoch. NoiseLoRA-SV resume checkpoints also use the same per-module coverage checks as full evaluation checkpoints. Baseline resume checkpoints require near-complete baseline model coverage, so encoder-only ECAPA files are rejected for `--resume`.

Saved training checkpoints store compact non-sensitive configuration metadata only: `audio`, `model`, `loss`, `optimizer`, `scheduler`, and `training`. Local dataset paths, checkpoint paths, output directories, and `_config_dir` are excluded.

## Operational Notes

- Numeric results can vary slightly across GPU types, dependency versions, random seeds, and dataset preparation details.
- Keep `training.babble_sources` and `evaluation.babble_sources` consistent with the provided experiment configuration when comparing Babble results.
- Parameter counts may differ if configurable channel sizes, CRN widths, LoRA rank, or hypernetwork dimensions are changed; the code reports these counts directly.
