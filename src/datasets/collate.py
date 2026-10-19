import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    # TODO
    spectra = []
    for item in dataset_items:
        spectrum = item["spectrogram"]
        if spectrum.ndim == 3 and spectrum.shape[0] == 1:
            spectrum = spectrum.squeeze(0)
        spectra.append(spectrum)
    lengths = torch.tensor(
        [spectrum.shape[-1] for spectrum in spectra], dtype=torch.long
    )
    max_time = int(lengths.max())
    spectra = torch.stack(
        [
            torch.nn.functional.pad(spectrum, (0, max_time - spectrum.shape[-1]))
            for spectrum in spectra
        ]
    )
    targets = [item["text_encoded"] for item in dataset_items]
    audio_lengths = torch.tensor(
        [item["audio"].shape[-1] for item in dataset_items], dtype=torch.long
    )
    max_audio = int(audio_lengths.max())
    audio = torch.stack(
        [
            torch.nn.functional.pad(
                item["audio"], (0, max_audio - item["audio"].shape[-1])
            )
            for item in dataset_items
        ]
    )
    return {
        "spectrogram": spectra,
        "spectrogram_length": lengths,
        "text_encoded": torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=0
        ),
        "text_encoded_length": torch.tensor(
            [target.numel() for target in targets], dtype=torch.long
        ),
        "text": [item["text"] for item in dataset_items],
        "audio": audio,
        "audio_length": audio_lengths,
        "audio_path": [item["audio_path"] for item in dataset_items],
    }


# ┌( ಠ_ಠ)┘🚬┌( ಠ_ಠ)┘🚬┌( ಠ_ಠ)┘🚬┌( ಠ_ಠ)┘🚬┌( ಠ_ಠ)┘🚬
