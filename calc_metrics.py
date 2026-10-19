import argparse
import json
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder import CTCTextEncoder


def calculate_metrics(ground_truth, predictions):
    ground_truth = Path(ground_truth)
    predictions = Path(predictions)

    references = {
        path.name: path for path in ground_truth.glob("*.txt") if path.is_file()
    }
    hypotheses = {
        path.name: path for path in predictions.glob("*.txt") if path.is_file()
    }
    if not references:
        raise ValueError("No reference .txt files found.")
    missing = references.keys() - hypotheses.keys()
    extra = hypotheses.keys() - references.keys()
    if missing or extra:
        raise ValueError(
            f"Missing prediction files: {sorted(missing)}; unexpected files: {sorted(extra)}"
        )

    character_errors = word_errors = 0
    characters = words = 0
    for name in sorted(references):
        reference = CTCTextEncoder.normalize_text(
            references[name].read_text(encoding="utf-8").strip()
        )
        prediction = CTCTextEncoder.normalize_text(
            hypotheses[name].read_text(encoding="utf-8").strip()
        )
        char_errors, char_count = calc_cer(reference, prediction)
        word_errs, word_count = calc_wer(reference, prediction)
        character_errors += char_errors
        characters += char_count
        word_errors += word_errs
        words += word_count

    return {
        "CER": character_errors / max(characters, 1),
        "WER": word_errors / max(words, 1),
        "samples": len(references),
        "character_errors": character_errors,
        "characters": characters,
        "word_errors": word_errors,
        "words": words,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    result = calculate_metrics(args.ground_truth, args.predictions)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
