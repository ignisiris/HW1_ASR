import torchaudio
from torch import nn


class LogMel(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_mels=128,
        n_fft=1024,
        win_length=400,
        hop_length=160,
        eps=1e-5,
    ):
        super().__init__()
        self.eps = eps
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode="constant",
        )

    def forward(self, audio):
        features = self.mel(audio).squeeze(0)
        features = features.clamp_min(self.eps).log()
        mean = features.mean(-1, keepdim=True)
        std = features.std(-1, keepdim=True, unbiased=False)
        return (features - mean) / std.clamp_min(self.eps)
