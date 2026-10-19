import torch  
from torch import nn


class TimeMasking(nn.Module):
    def __init__(self, max_ratio=0.05, max_width=40, num_masks=2, p=1.0):
        super().__init__()
        self.max_ratio = max_ratio
        self.max_width = max_width
        self.num_masks = num_masks
        self.p = p

    def forward(self, spectrum):
        if torch.rand(()) >= self.p:
            return spectrum
        result = spectrum.clone()
        time = result.shape[-1]
        maximum = min(self.max_width, int(time * self.max_ratio))
        for _ in range(self.num_masks):
            width = int(torch.randint(0, maximum + 1, ()))
            start = int(torch.randint(0, time - width + 1, ()))
            result[:, start:start + width] = 0
        return result
