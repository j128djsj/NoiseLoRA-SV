import numpy as np


def cosine_score(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def compute_eer(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0:
        raise ValueError("EER requires at least one score")
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("EER score and label lengths must match")
    # Reject invalid scores before threshold sorting.
    if not np.isfinite(scores).all():
        raise ValueError("EER scores must be finite.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("EER labels must be 0 or 1")
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("EER requires both positive and negative labels")
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    fars, frrs = [0.0], [1.0]
    accepted_pos, accepted_neg = 0, 0
    idx = 0
    while idx < sorted_scores.size:
        end = idx + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[idx]:
            end += 1
        group = sorted_labels[idx:end]
        # Update all samples sharing the same threshold together.
        accepted_pos += int(np.sum(group == 1))
        accepted_neg += int(np.sum(group == 0))
        fars.append(accepted_neg / negatives)
        frrs.append(1.0 - accepted_pos / positives)
        idx = end
    fars.append(1.0)
    frrs.append(0.0)
    fars = np.asarray(fars, dtype=np.float64)
    frrs = np.asarray(frrs, dtype=np.float64)
    diff = fars - frrs
    exact = np.where(np.isclose(diff, 0.0))[0]
    if exact.size:
        eer = 0.5 * (fars[exact[0]] + frrs[exact[0]])
        return float(eer * 100.0)
    for i in range(diff.size - 1):
        if diff[i] * diff[i + 1] < 0:
            t = -diff[i] / (diff[i + 1] - diff[i])
            eer = fars[i] + t * (fars[i + 1] - fars[i])
            return float(eer * 100.0)
    idx = int(np.argmin(np.abs(diff)))
    return float(0.5 * (fars[idx] + frrs[idx]) * 100.0)
