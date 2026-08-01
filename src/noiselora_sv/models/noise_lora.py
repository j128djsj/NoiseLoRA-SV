import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalNoiseConditionedLoRA(nn.Module):
    def __init__(self, channels, noise_dim, rank=4, alpha=8, hidden=256, init_scale=0.001):
        super().__init__()
        self.channels = int(channels)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.trunk = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_a = nn.Linear(hidden, self.rank * self.channels)
        self.head_b = nn.Linear(hidden, self.channels * self.rank)
        self.reset_parameters(float(init_scale))

    def reset_parameters(self, init_scale):
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.head_a.weight, std=init_scale)
        nn.init.normal_(self.head_b.weight, std=init_scale)
        nn.init.zeros_(self.head_a.bias)
        nn.init.zeros_(self.head_b.bias)

    def lora_residual(self, x, z_global):
        bsz, channels, _ = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")
        h = self.trunk(z_global)
        # Generate instance-specific low-rank matrices.
        a = self.head_a(h).view(bsz, self.rank, self.channels)
        b = self.head_b(h).view(bsz, self.channels, self.rank)
        # [B, C, T] -> [B, r, T]
        low = torch.einsum("brc,bct->brt", a, x.float())
        low = F.silu(low)
        return torch.einsum("bcr,brt->bct", b, low).to(x.dtype)

    def forward(self, x, z_global, z_local=None):
        return x + self.scale * self.lora_residual(x, z_global)


class HNCNoiseConditionedLoRA(GlobalNoiseConditionedLoRA):
    def __init__(self, channels, noise_dim, rank=4, alpha=8, hidden=256, gate_hidden=64, init_scale=0.001):
        super().__init__(channels, noise_dim, rank, alpha, hidden, init_scale)
        self.gate = nn.Sequential(
            nn.Conv2d(noise_dim, gate_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, kernel_size=1),
        )
        self.last_gate = None

    def temporal_gate(self, z_local, frames):
        if z_local is None:
            raise ValueError("HNC LoRA requires z_local")
        # Paper HNC gate: 3x3 Conv -> ReLU -> 1x1 Conv -> frequency pool -> Sigmoid.
        gate = torch.sigmoid(self.gate(z_local).mean(dim=2))
        if gate.size(-1) != frames:
            gate = F.interpolate(gate, size=frames, mode="linear", align_corners=False)
        self.last_gate = gate
        return gate

    def forward(self, x, z_global, z_local=None):
        residual = self.lora_residual(x, z_global)
        gate = self.temporal_gate(z_local, x.size(-1)).to(dtype=x.dtype)
        return x + self.scale * residual * gate
