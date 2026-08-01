import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmaxLoss(nn.Module):
    def __init__(self, n_classes, embedding_dim=192, margin=0.2, scale=30):
        super().__init__()
        self.margin = float(margin)
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(int(n_classes), int(embedding_dim)))
        nn.init.xavier_normal_(self.weight)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, embedding, label):
        label = label.long().to(embedding.device)
        cosine = F.linear(F.normalize(embedding.float(), dim=1), F.normalize(self.weight.float(), dim=1))
        cosine = cosine.clamp(-1.0 + 1e-5, 1.0 - 1e-5)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
        loss = F.cross_entropy(logits, label)
        acc = logits.argmax(dim=1).eq(label).float().mean() * 100.0
        return loss, acc, logits
