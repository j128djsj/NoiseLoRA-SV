# Third-Party Notices

## ECAPA-TDNN

The ECAPA-TDNN backbone implementation in
`models/ecapa_tdnn.py` is adapted from:

- Project: TaoRuijie/ECAPA-TDNN
- Source: https://github.com/TaoRuijie/ECAPA-TDNN
- Copyright: Copyright (c) 2022 Tao Ruijie
- License: MIT License

The upstream repository provides an implementation of the ECAPA-TDNN speaker
encoder and states that it is modified from `clovaai/voxceleb_trainer`.

The implementation in this repository has been refactored to expose the
`C`, `S1`, `S2`, and `S3` stages and to support the NoiseLoRA-SV adaptation
modules.

The NoiseLoRA-SV-specific components include:

- the CRN noise representation and reconstruction network;
- the Multi-Scale Noise Representation Head;
- MAGF multi-scale fusion;
- global noise-conditioned LoRA;
- hierarchical noise-conditioned LoRA;
- frame-level temporal gating;
- masked InfoNCE distillation;
- the teacher-student training pipeline;
- the NoiseLoRA-SV training and evaluation workflow.

The original MIT license notice is preserved in
`licenses/ECAPA-TDNN-MIT.txt`.
