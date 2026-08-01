import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedMaskedInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, eps=1e-8):
        super().__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, student_embedding, teacher_embedding, labels, valid=None):
        if teacher_embedding is None:
            raise ValueError("teacher_embedding is required when distillation is enabled")
        if valid is None:
            valid = torch.ones(student_embedding.size(0), device=student_embedding.device, dtype=torch.bool)
        else:
            valid = valid.to(student_embedding.device).bool()
        idx = valid.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return student_embedding.new_tensor(0.0)
        student = F.normalize(student_embedding.index_select(0, idx).float(), dim=1, eps=self.eps)
        teacher = F.normalize(teacher_embedding.index_select(0, idx).float(), dim=1, eps=self.eps)
        labels = labels.index_select(0, idx).to(student.device)
        logits = torch.matmul(student, teacher.t()) / max(self.temperature, 1e-6)
        same_speaker = labels.view(-1, 1).eq(labels.view(1, -1))
        positive = torch.eye(labels.numel(), device=student.device, dtype=torch.bool)
        # Mask same-speaker samples from the negative set.
        keep = positive | ~same_speaker
        logits = logits.masked_fill(~keep, torch.finfo(logits.dtype).min)
        target = torch.arange(labels.numel(), device=student.device)
        return F.cross_entropy(logits, target)
