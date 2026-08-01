# Pretrained Checkpoints

Place local ECAPA or NoiseLoRA-SV checkpoints here if desired. Model weights are not bundled in this repository.

Typical configuration fields:

- `paths.student_checkpoint`: clean ECAPA initialization for the student or baseline encoder.
- `paths.teacher_checkpoint`: independent clean ECAPA teacher checkpoint for distillation.
- `paths.checkpoint_dir`: output directory for full training checkpoints, defaulting to `outputs/checkpoints`.

Full resume checkpoints should be passed with `--resume`, not through the speaker-pretraining fields.
