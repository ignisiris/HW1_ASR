import argparse
import hashlib
import json
import pickle
from pathlib import Path

from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from src.text_encoder.word_lm import ModifiedKneserNeyTrigram
from src.utils.io_utils import ROOT_PATH

TRAIN_PARTS = ("train-clean-100", "train-clean-360", "train-other-500")


def train_lm(data_dir, output, parts=TRAIN_PARTS):
    if not parts or any(part not in TRAIN_PARTS for part in parts):
        raise ValueError("LM may be trained only on LibriSpeech training splits, bro")
    output = Path(output)
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(
            "Choose a new output path; existing LM is preserved, good work"
        )
    texts, sources = [], []
    for part in parts:
        content = (Path(data_dir) / (part + "_index.json")).read_bytes()
        records = json.loads(content)
        sources.append(
            dict(
                part=part,
                records=len(records),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
        texts.extend(
            CTCTextEncoder.normalize_text(record["text"]) for record in records
        )
    lm = ModifiedKneserNeyTrigram(texts)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        pickle.dump(lm, destination, protocol=4)
    metadata = dict(
        sources=sources,
        training_sentences=lm.training_sentences,
        training_words=lm.training_words,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT_PATH / "data/datasets/librispeech"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT_PATH / "saved/exp79/word_lm_train_all.pkl"
    )
    parser.add_argument(
        "--parts", nargs="+", choices=TRAIN_PARTS, default=list(TRAIN_PARTS)
    )
    args = parser.parse_args()
    print(json.dumps(train_lm(args.data_dir, args.output, args.parts), indent=2))


if __name__ == "__main__":
    main()
