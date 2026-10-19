from pathlib import Path

import torch
import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)


class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        data_dir = Path(data_dir)
        data = []
        ids = set()
        for path in sorted((data_dir / "audio").iterdir()):
            if not path.is_file() or path.suffix.lower() not in {
                ".wav",
                ".flac",
                ".mp3",
                ".m4a",
            }:
                continue
            if path.stem in ids:
                raise ValueError("Repeated audio ID: " + path.stem)
            ids.add(path.stem)
            info = torchaudio.info(str(path))
            transcript = data_dir / "transcriptions" / (path.stem + ".txt")
            data.append(
                {
                    "path": str(path.resolve()),
                    "audio_len": info.num_frames / info.sample_rate,
                    "text": transcript.read_text().strip()
                    if transcript.is_file()
                    else None,
                }
            )
        super().__init__(data, *args, **kwargs)

    def __getitem__(self, ind):
        item = self._index[ind]
        audio = self.load_audio(item["path"])
        if self.instance_transforms is not None and "audio" in self.instance_transforms:
            audio = self.instance_transforms["audio"](audio)
        text = item["text"]
        instance_data = {
            "audio": audio,
            "spectrogram": self.get_spectrogram(audio),
            "text": text,
            "text_encoded": self.text_encoder.encode(text)
            if text is not None
            else torch.empty(0, dtype=torch.long),
            "audio_path": item["path"],
        }
        return self.preprocess_data(instance_data)
