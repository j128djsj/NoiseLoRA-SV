import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_out(length, kernel=3, padding=1, stride=2, dilation=1):
    return (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


class ConvBNPReLU2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        return self.net(x)


class MAGF(nn.Module):
    def __init__(self, channels, scales=3):
        super().__init__()
        self.scales = int(scales)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * scales, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, scales, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(scales),
            nn.Sigmoid(),
        )

    def forward(self, features):
        stacked = torch.stack(features, dim=1)
        weights = self.gate(torch.cat(features, dim=1)).unsqueeze(2)
        # Fuse aligned multi-scale noise cues with learned gates.
        return (stacked * weights).sum(dim=1)


class MultiScaleNoiseRepresentationHead(nn.Module):
    def __init__(self, channels, noise_dim):
        super().__init__()
        c2, c3, c4 = channels
        self.global_head = nn.Sequential(
            nn.Conv2d(c4, noise_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(noise_dim, noise_dim, kernel_size=1),
        )
        self.proj2 = nn.Conv2d(c2, noise_dim, kernel_size=3, padding=1)
        self.proj3 = nn.Conv2d(c3, noise_dim, kernel_size=3, padding=1)
        self.proj4 = nn.Conv2d(c4, noise_dim, kernel_size=3, padding=1)
        self.magf = MAGF(noise_dim, scales=3)
        self.local_out = nn.Conv2d(noise_dim, noise_dim, kernel_size=1)

    @staticmethod
    def _align(x, size):
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, e2, e3, e4):
        # Extract global noise context from E4.
        z_global = self.global_head(e4).mean(dim=(2, 3))
        size = e2.shape[-2:]
        z2 = self.proj2(e2)
        z3 = self._align(self.proj3(e3), size)
        z4 = self._align(self.proj4(e4), size)
        z_local = self.local_out(self.magf([z2, z3, z4]))
        return z_global, z_local


class CRNNoiseReconstructionNetwork(nn.Module):
    def __init__(self, n_mels=80, base_channels=32, rnn_channels=64, rnn_hidden=256, noise_dim=128):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.E1 = nn.Sequential(ConvBNPReLU2d(1, c1), ConvBNPReLU2d(c1, c1, stride=(2, 1)))
        self.E2 = nn.Sequential(ConvBNPReLU2d(c1, c2), ConvBNPReLU2d(c2, c2, stride=(2, 1)))
        self.E3 = nn.Sequential(ConvBNPReLU2d(c2, c3), ConvBNPReLU2d(c3, c3, stride=(2, 1)))
        self.E4 = nn.Sequential(ConvBNPReLU2d(c3, c4), ConvBNPReLU2d(c4, c4, stride=(2, 1)))
        f4 = int(n_mels)
        for _ in range(4):
            f4 = _conv_out(f4)
        self.R = nn.Conv2d(c4, rnn_channels, kernel_size=1, bias=False)
        self.gru = nn.GRU(rnn_channels * f4, rnn_hidden, batch_first=True)
        self.gru_out = nn.Linear(rnn_hidden, rnn_channels * f4)
        self.f4 = f4
        self.rnn_channels = int(rnn_channels)
        self.D4_up = nn.ConvTranspose2d(rnn_channels, c3, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
        self.D4 = nn.Sequential(ConvBNPReLU2d(c3 + c3, c3), ConvBNPReLU2d(c3, c2))
        self.D3_up = nn.ConvTranspose2d(c2, c2, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
        self.D3 = nn.Sequential(ConvBNPReLU2d(c2 + c2, c2), ConvBNPReLU2d(c2, c1))
        self.D2_up = nn.ConvTranspose2d(c1, c1, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
        self.D2 = nn.Sequential(ConvBNPReLU2d(c1 + c1, c1), ConvBNPReLU2d(c1, c1))
        self.D1_up = nn.ConvTranspose2d(c1, c1, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
        self.D1 = nn.Conv2d(c1, 1, kernel_size=1)
        self.msnrh = MultiScaleNoiseRepresentationHead((c2, c3, c4), noise_dim)

    @staticmethod
    def _fix_size(x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, logmel):
        x = logmel.unsqueeze(1)
        e1 = self.E1(x)
        e2 = self.E2(e1)
        e3 = self.E3(e2)
        e4 = self.E4(e3)
        z_global, z_local = self.msnrh(e2, e3, e4)
        r = self.R(e4)
        bsz, channels, freq, frames = r.shape
        # [B, C, F, T] -> [B, T, C*F] for the CRN recurrent bottleneck.
        seq = r.permute(0, 3, 1, 2).reshape(bsz, frames, channels * freq)
        seq, _ = self.gru(seq)
        seq = self.gru_out(seq).view(bsz, frames, self.rnn_channels, self.f4)
        r = seq.permute(0, 2, 3, 1).contiguous()
        d4 = self.D4(torch.cat([self._fix_size(self.D4_up(r), e3), e3], dim=1))
        d3 = self.D3(torch.cat([self._fix_size(self.D3_up(d4), e2), e2], dim=1))
        d2 = self.D2(torch.cat([self._fix_size(self.D2_up(d3), e1), e1], dim=1))
        d1 = self._fix_size(self.D1_up(d2), x)
        noise_hat = self.D1(d1).squeeze(1)
        return {
            "noise_hat": noise_hat,
            "z_global": z_global,
            "z_local": z_local,
            "encoder_features": {"E1": e1, "E2": e2, "E3": e3, "E4": e4},
        }
