import pytest

from noiselora_sv.data.voxceleb import VoxCelebTrainDataset
from noiselora_sv.data.voxceleb import VoxCelebTrialsDataset


def test_train_list_speaker_then_relative_path(tmp_path):
    wav = tmp_path / "id10001" / "utt.wav"
    wav.parent.mkdir()
    wav.write_bytes(b"placeholder")
    train_list = tmp_path / "train.txt"
    train_list.write_text("id10001 id10001/utt.wav\n", encoding="utf-8")
    dataset = VoxCelebTrainDataset(train_root=str(tmp_path), train_list=str(train_list), augment_prob=0.0)
    assert len(dataset) == 1
    assert dataset.items[0][1] == 0


def test_train_list_malformed_line_raises(tmp_path):
    train_list = tmp_path / "train.txt"
    train_list.write_text("id10001 only two extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train list lines"):
        VoxCelebTrainDataset(train_root=str(tmp_path), train_list=str(train_list), augment_prob=0.0)


def test_baseline_config_disables_augmentation():
    text = __import__("pathlib").Path("configs/baseline_ecapa.yaml").read_text(encoding="utf-8")
    assert "augment_prob: 0.0" in text


def test_trial_line_malformed_raises(tmp_path):
    trials = tmp_path / "trials.txt"
    trials.write_text("1 only_one_path\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trial lines"):
        VoxCelebTrialsDataset(str(tmp_path), str(trials))


def test_trial_invalid_label_raises(tmp_path):
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    first.write_bytes(b"x")
    second.write_bytes(b"x")
    trials = tmp_path / "trials.txt"
    trials.write_text("2 a.wav b.wav\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label must be 0 or 1"):
        VoxCelebTrialsDataset(str(tmp_path), str(trials))


def test_trial_missing_audio_raises(tmp_path):
    first = tmp_path / "a.wav"
    first.write_bytes(b"x")
    trials = tmp_path / "trials.txt"
    trials.write_text("1 a.wav missing.wav\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="trial audio not found"):
        VoxCelebTrialsDataset(str(tmp_path), str(trials))
