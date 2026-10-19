import torch
from torch import nn


class FrequencyMasking(nn.Module):
    def __init__(self, max_width=12, num_masks=2, p=1.0):
        super().__init__()
        self.max_width = max_width
        self.num_masks = num_masks
        self.p = p

    def forward(self, spectrum):
        if torch.rand(()) >= self.p:
            return spectrum
        result = spectrum.clone()
        frequency = result.shape[0]
        for _ in range(self.num_masks):
            width = int(torch.randint(0, min(self.max_width, frequency) + 1, ()))
            start = int(torch.randint(0, frequency - width + 1, ()))
            result[start : start + width, :] = 0
        return result
