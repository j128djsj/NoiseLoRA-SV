import pytest

from utils.metrics import compute_eer


def test_eer_perfectly_separated_scores():
    assert compute_eer([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(0.0)


def test_eer_completely_reversed_scores():
    assert compute_eer([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(100.0)


def test_eer_identical_positive_and_negative_scores():
    assert compute_eer([0.5, 0.5], [1, 0]) == pytest.approx(50.0)


def test_eer_multiple_tied_score_groups():
    scores = [0.9, 0.9, 0.5, 0.5, 0.1, 0.1]
    labels = [1, 1, 1, 0, 0, 0]
    assert compute_eer(scores, labels) == pytest.approx(100.0 / 6.0)


def test_eer_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="positive and negative"):
        compute_eer([0.1, 0.2], [1, 1])
    with pytest.raises(ValueError, match="positive and negative"):
        compute_eer([0.1, 0.2], [0, 0])
    with pytest.raises(ValueError, match="at least one"):
        compute_eer([], [])
    with pytest.raises(ValueError, match="lengths"):
        compute_eer([0.1], [1, 0])


def test_eer_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        compute_eer([0.1, float("nan")], [1, 0])
    with pytest.raises(ValueError, match="finite"):
        compute_eer([0.1, float("inf")], [1, 0])
    with pytest.raises(ValueError, match="finite"):
        compute_eer([0.1, float("-inf")], [1, 0])
