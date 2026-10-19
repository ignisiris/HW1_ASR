import torch
import torchaudio
from torch import nn


class SpeedPerturbation(nn.Module):
    def __init__(self, sample_rate=16000, factors=(0.9, 1.0, 1.1)):
        super().__init__()
        self.transforms = nn.ModuleList(
            [
                torchaudio.transforms.Resample(
                    orig_freq=round(sample_rate * factor),
                    new_freq=sample_rate,
                )
                for factor in factors
            ]
        )

    def forward(self, audio):
        return self.transforms[int(torch.randint(len(self.transforms), ()))](audio)
