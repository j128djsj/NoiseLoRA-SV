from math import gcd

import numpy as np
import soundfile as sf
from scipy import signal


def resample_wav(wav, orig_sr, target_sr):
    if int(orig_sr) == int(target_sr):
        return wav.astype(np.float32)
    factor = gcd(int(orig_sr), int(target_sr))
    wav = signal.resample_poly(wav, int(target_sr) // factor, int(orig_sr) // factor)
    return wav.astype(np.float32)


def read_audio(path, sample_rate):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav[:, 0]
    wav = wav.astype(np.float32)
    return resample_wav(wav, sr, sample_rate)


def crop_or_pad(wav, length, random_crop=True, rng=None):
    if rng is None:
        rng = np.random
    length = int(length)
    if wav.size == 0:
        return np.zeros(length, dtype=np.float32)
    if len(wav) < length:
        reps = length // len(wav) + 1
        wav = np.tile(wav, reps)
    if len(wav) > length:
        start = rng.randint(0, len(wav) - length + 1) if random_crop else 0
        wav = wav[start:start + length]
    return wav.astype(np.float32)
