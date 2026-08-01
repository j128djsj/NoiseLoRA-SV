import glob
import hashlib
import os
import random

import numpy as np

from noiselora_sv.data.audio import read_audio


def stable_hash_int(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _mix_at_snr(clean, noise, snr_db, peak_limit=0.99, min_rms=1e-4, max_gain_db=40.0):
    clean = clean.astype(np.float32)
    noise = noise.astype(np.float32)
    clean_rms = np.sqrt(np.mean(clean ** 2) + min_rms ** 2)
    noise_rms = np.sqrt(np.mean(noise ** 2) + min_rms ** 2)
    gain_db = 20.0 * np.log10(clean_rms + 1e-12) - 20.0 * np.log10(noise_rms + 1e-12) - float(snr_db)
    gain = 10.0 ** (float(np.clip(gain_db, -max_gain_db, max_gain_db)) / 20.0)
    added = noise * gain
    clean_target = clean
    mixed = clean_target + added
    peak = float(np.max(np.abs(mixed)) + 1e-12)
    if peak > peak_limit:
        scale = peak_limit / peak
        clean_target = clean_target * scale
        added = added * scale
        # Preserve mixed = clean + noise after peak scaling.
        mixed = clean_target + added
    if not np.isfinite(mixed).all():
        return clean, clean, np.zeros_like(clean, dtype=np.float32), 0.0
    return mixed.astype(np.float32), clean_target.astype(np.float32), added.astype(np.float32), 1.0


class NoiseMixer:
    def __init__(self, noise_root, sample_rate=16000, snr_range=(0.0, 20.0), exts=("wav", "flac"), required=False):
        self.noise_root = noise_root or ""
        self.sample_rate = int(sample_rate)
        self.snr_range = tuple(float(x) for x in snr_range)
        if required and (not self.noise_root or not os.path.isdir(self.noise_root)):
            raise FileNotFoundError(f"Noise root is required and was not found: {self.noise_root}")
        self.files = []
        if self.noise_root and os.path.isdir(self.noise_root):
            for ext in exts:
                self.files.extend(glob.glob(os.path.join(self.noise_root, "**", f"*.{ext}"), recursive=True))
        self.files = sorted(self.files)
        if required and not self.files:
            raise FileNotFoundError(f"No noise audio files found under: {self.noise_root}")

    def _read_segment(self, length, rng):
        if not self.files:
            return None
        wav = read_audio(rng.choice(self.files), self.sample_rate)
        if len(wav) < length:
            wav = np.tile(wav, length // len(wav) + 1)
        start = rng.randint(0, len(wav) - length) if len(wav) > length else 0
        return wav[start:start + length].astype(np.float32)

    def add(self, clean, rng=None, snr_db=None):
        rng = rng or random
        noise = self._read_segment(len(clean), rng)
        if noise is None:
            return clean.astype(np.float32), clean.astype(np.float32), np.zeros_like(clean, dtype=np.float32), 0.0
        if snr_db is None:
            snr_db = rng.uniform(*self.snr_range)
        return _mix_at_snr(clean, noise, snr_db)


class MusanNoiseMixer:
    def __init__(
        self,
        musan_root,
        sample_rate=16000,
        snr_range=(0.0, 20.0),
        noise_types=None,
        babble_sources=3,
        required=False,
    ):
        self.sample_rate = int(sample_rate)
        self.snr_range = tuple(float(x) for x in snr_range)
        self.noise_types = list(noise_types or ["babble", "music", "noise"])
        self.babble_sources = int(babble_sources)
        self.root = musan_root or ""
        if required and (not self.root or not os.path.isdir(self.root)):
            raise FileNotFoundError(f"MUSAN split root is required and was not found: {self.root}")
        self.files = {}
        for name in ["noise", "speech", "music"]:
            pattern = os.path.join(self.root, name, "**", "*.wav") if self.root else ""
            self.files[name] = sorted(glob.glob(pattern, recursive=True)) if pattern else []
        if required:
            self._validate_required_types(self.noise_types)

    def _validate_required_types(self, noise_types):
        for name in noise_types:
            source = self._source_name(name)
            if source == "speech" and name == "babble" and self.babble_sources < 1:
                raise ValueError("babble_sources must be >= 1")
            if not self.files.get(source):
                raise FileNotFoundError(f"Required MUSAN '{source}' wav files are missing under: {self.root}")

    @staticmethod
    def _source_name(noise_type):
        mapping = {"babble": "speech", "speech": "speech", "music": "music", "noise": "noise"}
        if noise_type not in mapping:
            raise ValueError(f"Unsupported MUSAN noise type: {noise_type}")
        return mapping[noise_type]

    def _pick_type(self, rng):
        available = [name for name in self.noise_types if self.files.get(self._source_name(name))]
        return rng.choice(available) if available else None

    def _segment(self, source, length, rng):
        files = self.files.get(source, [])
        if not files:
            return None
        wav = read_audio(rng.choice(files), self.sample_rate)
        if len(wav) < length:
            wav = np.tile(wav, length // len(wav) + 1)
        start = rng.randint(0, len(wav) - length) if len(wav) > length else 0
        return wav[start:start + length].astype(np.float32)

    def _babble(self, length, rng):
        parts = []
        for _ in range(max(1, self.babble_sources)):
            segment = self._segment("speech", length, rng)
            if segment is not None:
                parts.append(segment)
        if not parts:
            return None
        # Synthesize Babble by mixing multiple speech noise sources.
        return np.mean(np.stack(parts, axis=0), axis=0).astype(np.float32)

    def add(self, clean, rng=None, snr_db=None, noise_type=None):
        rng = rng or random
        noise_type = noise_type or self._pick_type(rng)
        if not noise_type:
            return clean.astype(np.float32), clean.astype(np.float32), np.zeros_like(clean, dtype=np.float32), 0.0
        if noise_type == "babble":
            noise = self._babble(len(clean), rng)
        else:
            noise = self._segment(self._source_name(noise_type), len(clean), rng)
        if noise is None:
            return clean.astype(np.float32), clean.astype(np.float32), np.zeros_like(clean, dtype=np.float32), 0.0
        if snr_db is None:
            snr_db = rng.uniform(*self.snr_range)
        return _mix_at_snr(clean, noise, snr_db)
