import torch
from torch import nn


class AddNoise(nn.Module):
    def __init__(self, min_snr_db=15.0, max_snr_db=30.0, p=0.5):
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p

    def forward(self, audio):
        if torch.rand(()) >= self.p:
            return audio
        noise = torch.randn_like(audio)
        signal_rms = audio.square().mean(-1, keepdim=True).sqrt()
        noise_rms = noise.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        snr = torch.empty((), device=audio.device).uniform_(
            self.min_snr_db, self.max_snr_db
        )
        scale = signal_rms / (noise_rms * 10 ** (snr / 20))
        return audio + noise * scale
