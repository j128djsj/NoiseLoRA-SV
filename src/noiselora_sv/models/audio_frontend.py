import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class PreEmphasis(nn.Module):
    def __init__(self, coef=0.97):
        super().__init__()
        self.register_buffer("kernel", torch.tensor([-coef, 1.0]).view(1, 1, 2), persistent=False)

    def forward(self, wav):
        wav = wav.unsqueeze(1)
        wav = F.pad(wav, (1, 0), mode="reflect")
        return F.conv1d(wav, self.kernel).squeeze(1)


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        super().__init__()
        self.freq_mask_width = tuple(freq_mask_width)
        self.time_mask_width = tuple(time_mask_width)

    def _mask(self, x, dim, width_range):
        lo, hi = int(width_range[0]), int(width_range[1])
        if hi <= lo:
            return x
        bsz, n_mels, frames = x.shape
        size = n_mels if dim == 1 else frames
        max_width = min(hi, size)
        if max_width <= 0:
            return x
        widths = torch.randint(lo, max_width + 1, (bsz,), device=x.device)
        starts = torch.stack([
            torch.randint(0, max(1, size - int(width) + 1), (1,), device=x.device).squeeze(0)
            for width in widths
        ])
        axis = torch.arange(size, device=x.device).view(1, -1)
        mask = (axis >= starts.view(-1, 1)) & (axis < (starts + widths).view(-1, 1))
        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
        return x.masked_fill(mask, 0.0)

    def forward(self, x):
        x = self._mask(x, 2, self.time_mask_width)
        x = self._mask(x, 1, self.freq_mask_width)
        return x


class LogMelFrontend(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        sample_rate = int(cfg.get("sample_rate", 16000))
        win_length = int(round(sample_rate * float(cfg.get("win_length_ms", 25)) / 1000.0))
        hop_length = int(round(sample_rate * float(cfg.get("hop_length_ms", 10)) / 1000.0))
        self.n_mels = int(cfg.get("n_mels", 80))
        self.log_eps = float(cfg.get("log_eps", 1e-6))
        self.mean_norm = bool(cfg.get("mean_norm", True))
        self.preemphasis = PreEmphasis(float(cfg.get("preemphasis", 0.97)))
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(cfg.get("n_fft", 512)),
            win_length=win_length,
            hop_length=hop_length,
            f_min=float(cfg.get("f_min", 20.0)),
            f_max=float(cfg.get("f_max", 7600.0)),
            window_fn=torch.hamming_window,
            n_mels=self.n_mels,
        )
        aug_cfg = cfg.get("specaugment", {})
        self.specaugment_enabled = bool(aug_cfg.get("enabled", True))
        self.specaugment = SpecAugment(
            aug_cfg.get("freq_mask_width", (0, 8)),
            aug_cfg.get("time_mask_width", (0, 10)),
        )

    def forward(self, wav, aug=False):
        if wav.dim() == 3 and wav.size(1) == 1:
            wav = wav[:, 0]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        # Use one shared log-Mel frontend for student, teacher, and noise targets.
        x = self.preemphasis(wav.float())
        x = self.melspec(x).clamp_min(self.log_eps).log()
        if self.mean_norm:
            x = x - x.mean(dim=-1, keepdim=True)
        if aug and self.training and self.specaugment_enabled:
            x = self.specaugment(x)
        return x
