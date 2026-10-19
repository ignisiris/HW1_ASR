import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CNNGRUCTC(nn.Module):
    def __init__(
        self,
        n_feats,
        n_tokens,
        fc_hidden=512,
        rnn_hidden=256,
        rnn_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.ReLU(),
        )
        frequency = (n_feats + 1) // 2
        frequency = (frequency + 1) // 2
        self.projection = nn.Sequential(
            nn.Linear(64 * frequency, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.rnn = nn.GRU(
            input_size=fc_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden * 2, n_tokens),
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        output = self.conv1(spectrogram.unsqueeze(1))
        mask = (
            torch.arange(output.shape[-1], device=output.device)[None, :]
            < log_probs_length.to(output.device)[:, None]
        )
        output = output * mask[:, None, None, :]
        output = self.conv2(output) * mask[:, None, None, :]
        output = output.permute(0, 3, 1, 2).flatten(2)
        output = self.projection(output) * mask[:, :, None]
        packed = pack_padded_sequence(
            output,
            log_probs_length.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(
            packed, batch_first=True, total_length=output.shape[1]
        )
        output = self.classifier(output)
        log_probs = nn.functional.log_softmax(output, dim=-1)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return (input_lengths + 1) // 2

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        return result_info
