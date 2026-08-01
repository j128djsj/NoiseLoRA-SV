import numpy as np
import pytest

from datasets.noise import MusanNoiseMixer, _mix_at_snr


def test_peak_scaled_mix_is_clean_plus_noise():
    clean = np.ones(1600, dtype=np.float32)
    noise = np.ones(1600, dtype=np.float32)
    mixed, clean_target, noise_target, valid = _mix_at_snr(clean, noise, snr_db=0)
    assert valid == 1.0
    assert np.allclose(mixed, clean_target + noise_target, atol=1e-6)


def test_required_musan_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MusanNoiseMixer(tmp_path / "missing", required=True)
