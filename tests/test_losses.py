import pytest
import torch
import torch.nn.functional as F

from noiselora_sv.losses import AAMSoftmaxLoss, NoiseLoRASVLoss, SupervisedMaskedInfoNCELoss


def test_aam_softmax_is_finite():
    loss_fn = AAMSoftmaxLoss(n_classes=3, embedding_dim=192)
    embeddings = torch.randn(4, 192)
    labels = torch.tensor([0, 1, 2, 1])
    loss, acc, logits = loss_fn(embeddings, labels)
    assert torch.isfinite(loss)
    assert torch.isfinite(acc)
    assert logits.shape == (4, 3)


def test_supervised_infonce_masks_same_speaker_negatives():
    loss_fn = SupervisedMaskedInfoNCELoss(temperature=1.0)
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    teacher = student.clone()
    labels = torch.tensor([0, 1, 0])
    loss = loss_fn(student, teacher, labels)
    logits = F.normalize(student, dim=1) @ F.normalize(teacher, dim=1).t()
    keep = torch.eye(3, dtype=torch.bool) | ~labels.view(-1, 1).eq(labels.view(1, -1))
    manual = F.cross_entropy(logits.masked_fill(~keep, torch.finfo(logits.dtype).min), torch.arange(3))
    assert torch.allclose(loss, manual)


def test_distillation_requires_teacher_embedding_when_enabled():
    loss_fn = NoiseLoRASVLoss({"n_classes": 3, "distill_weight": 1.0, "noise_weight": 0.0}, embedding_dim=192)
    outputs = {"embedding": torch.randn(2, 192)}
    with pytest.raises(ValueError, match="teacher_embedding"):
        loss_fn(outputs, torch.tensor([0, 1]))


def test_noise_loss_requires_target_when_enabled():
    loss_fn = NoiseLoRASVLoss({"n_classes": 3, "distill_weight": 0.0, "noise_weight": 0.1}, embedding_dim=192)
    outputs = {"embedding": torch.randn(2, 192), "noise_hat": torch.randn(2, 80, 10)}
    with pytest.raises(ValueError, match="noise_hat and noise_target"):
        loss_fn(outputs, torch.tensor([0, 1]))
