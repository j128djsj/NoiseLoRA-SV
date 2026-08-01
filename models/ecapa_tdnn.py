# Adapted from TaoRuijie/ECAPA-TDNN:
# https://github.com/TaoRuijie/ECAPA-TDNN
# Copyright (c) 2022 Tao Ruijie
# Licensed under the MIT License.

import math

import torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class SERes2Block(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=2, scale=8):
        super().__init__()
        width = int(math.floor(channels / scale))
        self.width = width
        self.scale = scale
        self.conv1 = nn.Conv1d(channels, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size, padding=(kernel_size // 2) * dilation, dilation=dilation)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(scale - 1)])
        self.conv3 = nn.Conv1d(width * scale, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.se = SEModule(channels)

    def forward(self, x):
        residual = x
        out = self.bn1(self.relu(self.conv1(x)))
        chunks = torch.split(out, self.width, dim=1)
        outputs = []
        running = None
        for idx, conv in enumerate(self.convs):
            running = chunks[idx] if running is None else running + chunks[idx]
            running = self.bns[idx](self.relu(conv(running)))
            outputs.append(running)
        outputs.append(chunks[-1])
        out = torch.cat(outputs, dim=1)
        out = self.bn3(self.relu(self.conv3(out)))
        return self.se(out) + residual


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, hidden=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, hidden, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Tanh(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        x = x.float()
        frames = x.size(-1)
        mean = x.mean(dim=2, keepdim=True).repeat(1, 1, frames)
        std = x.var(dim=2, keepdim=True).clamp_min(1e-4).sqrt().repeat(1, 1, frames)
        weights = self.attention(torch.cat([x, mean, std], dim=1))
        mu = torch.sum(x * weights, dim=2)
        sigma = torch.sum((x ** 2) * weights, dim=2).sub(mu ** 2).clamp_min(1e-4).sqrt()
        return torch.cat([mu, sigma], dim=1)


class ECAPATDNN(nn.Module):
    def __init__(self, channels=1024, embedding_dim=192):
        super().__init__()
        self.channels = int(channels)
        self.embedding_dim = int(embedding_dim)
        self.C = nn.Sequential(
            nn.Conv1d(80, self.channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(self.channels),
        )
        self.S1 = SERes2Block(self.channels, kernel_size=3, dilation=2)
        self.S2 = SERes2Block(self.channels, kernel_size=3, dilation=3)
        self.S3 = SERes2Block(self.channels, kernel_size=3, dilation=4)
        self.mfa = nn.Sequential(nn.Conv1d(self.channels * 3, 1536, kernel_size=1), nn.ReLU())
        self.pool = AttentiveStatisticsPooling(1536)
        self.bn = nn.BatchNorm1d(3072)
        self.linear = nn.Linear(3072, self.embedding_dim)
        self.out_bn = nn.BatchNorm1d(self.embedding_dim)

    @staticmethod
    def _adapt(stage, x, adapters, z_global, z_local):
        if adapters is None or stage not in adapters:
            return x
        return adapters[stage](x, z_global, z_local)

    def forward_logmel(self, logmel, adapters=None, z_global=None, z_local=None, return_features=False):
        x = self.C(logmel)
        x = self._adapt("C", x, adapters, z_global, z_local)
        x1 = self.S1(x)
        x1 = self._adapt("S1", x1, adapters, z_global, z_local)
        # ECAPA residual aggregation follows the paper stage order.
        x2 = self.S2(x + x1)
        x2 = self._adapt("S2", x2, adapters, z_global, z_local)
        x3 = self.S3(x + x1 + x2)
        x3 = self._adapt("S3", x3, adapters, z_global, z_local)
        seq = self.mfa(torch.cat([x1, x2, x3], dim=1))
        pooled = self.pool(seq)
        emb = self.out_bn(self.linear(self.bn(pooled)))
        if return_features:
            return emb, {"C": x, "S1": x1, "S2": x2, "S3": x3, "seq": seq}
        return emb
