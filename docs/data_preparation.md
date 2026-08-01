# Data Preparation

This document describes the datasets, directory layouts, list formats, and
checkpoint preparation required by the official NoiseLoRA-SV implementation.

The repository does not redistribute speech, noise, or pretrained model files.
Download each dataset from its official source and follow its original license
or terms of use.

## Dataset Overview

| Dataset or artifact | Purpose | Usage |
| --- | --- | --- |
| VoxCeleb1 | Clean speaker speech | Train the clean ECAPA-TDNN baseline and evaluate speaker verification |
| MUSAN | Seen-noise corpus | Create noisy training samples and seen-noise evaluation conditions |
| NonSpeech100 | Unseen-noise corpus | Evaluate out-of-domain noise robustness |
| Clean ECAPA checkpoint | Student and teacher initialization | Initialize two independent ECAPA-TDNN models before NoiseLoRA-SV training |

## 1. VoxCeleb1

VoxCeleb1 provides the clean speech used for speaker training and verification.

The expected setup is:

- VoxCeleb1 development split for training;
- VoxCeleb1 test split for verification;
- 1,211 training speakers, matching `loss.n_classes: 1211`;
- a verification trial file containing positive and negative utterance pairs.

The dataset is not included in this repository. Obtain it from the official
VoxCeleb source and follow its access requirements.

### Expected layout

```text
VoxCeleb1/
├── dev/
│   └── wav/
│       ├── id10001/
│       │   └── <video_id>/
│       │       └── *.wav
│       ├── id10002/
│       └── ...
└── test/
    └── wav/
        ├── id10270/
        │   └── <video_id>/
        │       └── *.wav
        └── ...
```

### Training data

Set:

```yaml
paths:
  train_root: /path/to/VoxCeleb1/dev/wav
  train_list: ""
```

When `train_list` is empty, the loader scans:

```text
train_root/<speaker_id>/**/*.wav
```

You may also provide an explicit training list:

```text
<speaker_id> <relative_audio_path>
```

Example:

```text
id10001 id10001/1zcIwhmdeo4/00001.wav
id10002 id10002/6WO410QOeuo/00003.wav
```

Paths are resolved relative to `paths.train_root`.

### Verification data

Set:

```yaml
paths:
  test_root: /path/to/VoxCeleb1/test/wav
  trials: /path/to/voxceleb1_trials.txt
```

The supported trial format is:

```text
<label> <enroll_relative_audio_path> <test_relative_audio_path>
```

Example:

```text
1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6SegY/00008.wav
0 id10270/x6uYqmx31kE/00001.wav id10300/abc123/00002.wav
```

- `1` means the two utterances belong to the same speaker.
- `0` means the two utterances belong to different speakers.
- Both paths are resolved relative to `paths.test_root`.

Use the same trial file as the final experiment when reproducing the reported
EER values. Different VoxCeleb1 trial variants may produce different results.

## 2. MUSAN

MUSAN is used as the seen-noise corpus.

The original corpus contains three categories:

- `speech`;
- `music`;
- `noise`.

NoiseLoRA-SV uses them as:

| MUSAN category | Evaluation condition |
| --- | --- |
| `speech` | Babble |
| `music` | Music |
| `noise` | Noise |

Babble is synthesized by mixing multiple speech noise sources. The number of
speech sources is controlled by the configuration and should match the setting
used in the target experiment.

### Disjoint train and test splits

Training and evaluation noise files must be disjoint.

The code requires two explicit roots:

```yaml
paths:
  musan_train_root: /path/to/MUSAN/train
  musan_test_root: /path/to/MUSAN/test
```

Do not point both fields to the same directory.

### Expected layout

```text
MUSAN/
├── train/
│   ├── speech/
│   │   └── **/*.wav
│   ├── music/
│   │   └── **/*.wav
│   └── noise/
│       └── **/*.wav
└── test/
    ├── speech/
    │   └── **/*.wav
    ├── music/
    │   └── **/*.wav
    └── noise/
        └── **/*.wav
```

The loader searches recursively, so nested MUSAN subdirectories are allowed.

### Split preparation

Prepare the split before training:

1. Enumerate all MUSAN audio files.
2. Assign each file to either the training or test split.
3. Ensure no file appears in both splits.
4. Preserve the `speech`, `music`, and `noise` category structure.
5. Use the same split for all experiments that are compared.

For exact paper-result reproduction, use the original experiment split
manifests when available. Do not replace them with a new random split without
reporting the change.

## 3. NonSpeech100

NonSpeech100, also known as PNL 100 Nonspeech Sounds, is used only for
unseen-noise evaluation.

It is not used during training.

Set:

```yaml
paths:
  unseen_noise_root: /path/to/NonSpeech100
```

The loader searches recursively for supported audio files, including `.wav`
and `.flac`.

Example:

```text
NonSpeech100/
├── noise_001.wav
├── noise_002.wav
├── indoor/
│   └── *.wav
└── outdoor/
    └── *.wav
```

Audio is resampled to the configured sample rate at runtime.

Use the original dataset source or publication. Do not rely on an unverified
third-party mirror.

## 4. Clean ECAPA-TDNN Checkpoint

NoiseLoRA-SV uses an independent student and teacher model.

Both can be initialized from the same clean ECAPA-TDNN checkpoint, but they are
created as separate model instances. The teacher is frozen during
NoiseLoRA-SV training.

### Step 1: train the clean baseline

```bash
python main.py --config configs/baseline_ecapa.yaml --mode train
```

The baseline configuration uses clean VoxCeleb1 speech without MUSAN
augmentation.

### Step 2: select the clean checkpoint

After training, select the desired clean ECAPA-TDNN checkpoint.

### Step 3: configure NoiseLoRA-SV

```yaml
paths:
  student_checkpoint: /path/to/clean_ecapa_checkpoint.pth
  teacher_checkpoint: /path/to/clean_ecapa_checkpoint.pth
```

The student remains trainable. The teacher remains frozen and provides clean
speaker embeddings for masked InfoNCE distillation.

## 5. Complete Local Directory Example

A complete local dataset layout may look like:

```text
datasets-local/
├── VoxCeleb1/
│   ├── dev/
│   │   └── wav/
│   │       └── id10001/...
│   └── test/
│       └── wav/
│           └── id10270/...
├── MUSAN/
│   ├── train/
│   │   ├── speech/...
│   │   ├── music/...
│   │   └── noise/...
│   └── test/
│       ├── speech/...
│       ├── music/...
│       └── noise/...
└── NonSpeech100/
    └── **/*.wav
```

## 6. Configuration Example

```yaml
paths:
  # VoxCeleb1 development wav root.
  train_root: /path/to/VoxCeleb1/dev/wav

  # Optional: <speaker_id> <relative_audio_path>.
  train_list: ""

  # VoxCeleb1 test wav root.
  test_root: /path/to/VoxCeleb1/test/wav

  # VoxCeleb1 verification trial list.
  trials: /path/to/voxceleb1_trials.txt

  # Disjoint MUSAN split roots.
  musan_train_root: /path/to/MUSAN/train
  musan_test_root: /path/to/MUSAN/test

  # NonSpeech100 root for unseen-noise evaluation.
  unseen_noise_root: /path/to/NonSpeech100

  # Clean ECAPA initialization.
  student_checkpoint: /path/to/clean_ecapa_checkpoint.pth
  teacher_checkpoint: /path/to/clean_ecapa_checkpoint.pth
```

## 7. Audio Settings

The paper configuration uses:

```yaml
audio:
  sample_rate: 16000
  duration: 3.0
  n_fft: 512
  win_length_ms: 25
  hop_length_ms: 10
  n_mels: 80
```

The implementation requires `n_mels: 80` to match the ECAPA-TDNN input layer
used in the paper.

## 8. Evaluation Conditions

Seen-noise evaluation uses the MUSAN test split:

```text
Babble / Music / Noise
```

Unseen-noise evaluation uses NonSpeech100.

The configured SNR levels are:

```text
0, 5, 10, 15, 20 dB
```

Examples:

```bash
python main.py   --config configs/noiselora_ecapa.yaml   --mode eval   --condition clean   --checkpoint /path/to/noiselora_sv.pth
```

```bash
python main.py   --config configs/noiselora_ecapa.yaml   --mode eval   --condition seen   --noise-type babble   --snr 0   --checkpoint /path/to/noiselora_sv.pth
```

```bash
python main.py   --config configs/noiselora_ecapa.yaml   --mode eval   --condition unseen   --snr 0   --checkpoint /path/to/noiselora_sv.pth
```

## 9. Reproducibility Notes

For consistent results:

- keep the VoxCeleb1 trial file fixed;
- keep MUSAN training and test noise files disjoint;
- keep the MUSAN split unchanged across compared systems;
- keep the Babble source count fixed;
- use the same clean ECAPA initialization;
- use the same sample rate, crop duration, and Log-Mel frontend;
- record dependency versions and random seeds.

The exact trial list and MUSAN split used for a result should be documented
with the experiment.

## 10. Dataset Terms

Users are responsible for:

- obtaining the datasets from their original sources;
- following the original licenses and access terms;
- confirming whether trial files or split manifests may be redistributed;
- avoiding redistribution of dataset audio through this repository.

NoiseLoRA-SV does not claim ownership of VoxCeleb1, MUSAN, or NonSpeech100.
