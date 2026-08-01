import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from noiselora_sv.data.audio import crop_or_pad, read_audio
from noiselora_sv.data.noise import MusanNoiseMixer, NoiseMixer, stable_hash_int


def _parse_train_line(line):
    parts = line.strip().split()
    if len(parts) != 2:
        raise ValueError("train list lines must be: <speaker_id> <relative_audio_path>")
    return parts[1], parts[0]


class VoxCelebTrainDataset(Dataset):
    def __init__(
        self,
        train_root="",
        train_list="",
        sample_rate=16000,
        crop_seconds=3.0,
        musan_train_root="",
        augment_prob=0.6,
        snr_range=(0.0, 20.0),
        noise_types=None,
        babble_sources=3,
    ):
        self.train_root = train_root or ""
        self.train_list = train_list or ""
        self.sample_rate = int(sample_rate)
        self.crop_size = int(round(float(crop_seconds) * self.sample_rate))
        self.augment_prob = float(augment_prob)
        self.mixer = MusanNoiseMixer(
            musan_train_root,
            self.sample_rate,
            snr_range,
            noise_types=noise_types,
            babble_sources=babble_sources,
            required=self.augment_prob > 0,
        )
        self.items, self.utt_per_spk = self._load_items()

    def _load_items(self):
        raw = []
        if self.train_list and os.path.isfile(self.train_list):
            with open(self.train_list, "r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        parsed = _parse_train_line(line)
                    except ValueError as exc:
                        raise ValueError(f"{self.train_list}:{line_no}: {exc}") from exc
                    if parsed:
                        raw.append(parsed)
        elif self.train_root and os.path.isdir(self.train_root):
            for speaker in sorted(os.listdir(self.train_root)):
                spk_dir = os.path.join(self.train_root, speaker)
                if not os.path.isdir(spk_dir):
                    continue
                for root, _, files in os.walk(spk_dir):
                    for name in files:
                        if name.lower().endswith((".wav", ".flac")):
                            raw.append((os.path.join(root, name), speaker))
        labels = {spk: idx for idx, spk in enumerate(sorted({spk for _, spk in raw}))}
        items, utt_per_spk = [], {idx: [] for idx in labels.values()}
        for path, spk in raw:
            full_path = path if os.path.isabs(path) else os.path.join(self.train_root, path)
            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"Training audio not found for speaker {spk}: {full_path}")
            # Train lists are speaker-id first, then relative audio path.
            label = labels[spk]
            utt_per_spk[label].append(len(items))
            items.append((full_path, label))
        return items, utt_per_spk

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        clean = read_audio(path, self.sample_rate)
        clean = crop_or_pad(clean, self.crop_size, random_crop=True)
        noisy = clean
        clean_target = clean
        noise_target = np.zeros_like(clean, dtype=np.float32)
        noise_valid = 0.0
        if random.random() < self.augment_prob:
            noisy, clean_target, noise_target, noise_valid = self.mixer.add(clean)
        return {
            "noisy": torch.from_numpy(noisy),
            "clean": torch.from_numpy(clean_target),
            "clean_source": torch.from_numpy(clean),
            "label": torch.tensor(label, dtype=torch.long),
            "noise_target": torch.from_numpy(noise_target),
            "noise_valid": torch.tensor(noise_valid, dtype=torch.float32),
        }


class VoxCelebTrialsDataset(Dataset):
    def __init__(
        self,
        test_root,
        trials_path,
        sample_rate=16000,
        condition="clean",
        musan_test_root="",
        unseen_noise_root="",
        snr_db=None,
        noise_type=None,
        babble_sources=3,
        seed=0,
    ):
        self.test_root = test_root or ""
        self.trials_path = trials_path or ""
        self.sample_rate = int(sample_rate)
        self.condition = condition
        self.snr_db = snr_db
        self.noise_type = noise_type
        self.seed = int(seed)
        self.trials, self.path_list = self._parse_trials()
        self.seen_mixer = None
        self.unseen_mixer = None
        if self.condition == "seen":
            if self.noise_type is None:
                raise ValueError("condition='seen' requires a noise_type")
            # Use the explicit MUSAN test split for seen-noise EER.
            self.seen_mixer = MusanNoiseMixer(
                musan_test_root,
                self.sample_rate,
                (snr_db or 0.0, snr_db or 0.0),
                noise_types=[self.noise_type],
                babble_sources=babble_sources,
                required=True,
            )
        elif self.condition == "unseen":
            self.unseen_mixer = NoiseMixer(
                unseen_noise_root,
                self.sample_rate,
                (snr_db or 0.0, snr_db or 0.0),
                required=True,
            )

    def _parse_trials(self):
        trials, paths = [], set()
        if not self.trials_path or not os.path.isfile(self.trials_path):
            return trials, []
        with open(self.trials_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                parts = line.strip().split()
                if len(parts) != 3:
                    raise ValueError(f"{self.trials_path}:{line_no}: trial lines must be <label> <path1> <path2>")
                try:
                    label = int(parts[0])
                except ValueError as exc:
                    raise ValueError(f"{self.trials_path}:{line_no}: label must be 0 or 1") from exc
                if label not in (0, 1):
                    raise ValueError(f"{self.trials_path}:{line_no}: label must be 0 or 1")
                first, second = parts[1], parts[2]
                if not first or not second:
                    raise ValueError(f"{self.trials_path}:{line_no}: trial paths must be non-empty")
                for rel_path in (first, second):
                    full_path = os.path.join(self.test_root, rel_path)
                    if not os.path.isfile(full_path):
                        raise FileNotFoundError(f"{self.trials_path}:{line_no}: trial audio not found: {full_path}")
                trials.append((label, first, second))
                paths.update([first, second])
        return trials, sorted(paths)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        rel_path = self.path_list[index]
        wav = read_audio(os.path.join(self.test_root, rel_path), self.sample_rate)
        rng = random.Random(stable_hash_int(rel_path) + self.seed)
        if self.condition == "seen":
            wav, _, _, _ = self.seen_mixer.add(wav, rng=rng, snr_db=self.snr_db, noise_type=self.noise_type)
        elif self.condition == "unseen":
            wav, _, _, _ = self.unseen_mixer.add(wav, rng=rng, snr_db=self.snr_db)
        if not np.isfinite(wav).all():
            wav = np.zeros(self.sample_rate, dtype=np.float32)
        return torch.from_numpy(wav.astype(np.float32)), rel_path
