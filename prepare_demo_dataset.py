import argparse
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from download_resources import download
from src.utils.io_utils import ROOT_PATH


def archive_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid archive path: " + name)
    return root.joinpath(*relative.parts)


def prepare_dataset(url, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        temporary = Path(temporary)
        archive, extracted = temporary / "dataset", temporary / "extracted"
        extracted.mkdir()
        download(url, archive)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    archive_path(extracted, member.filename)
                source.extractall(extracted)
        else:
            with tarfile.open(archive) as source:
                for member in source.getmembers():
                    archive_path(extracted, member.name)
                    if not (member.isdir() or member.isfile()):
                        raise ValueError("Unsupported file in archive.")
                source.extractall(extracted)
        candidates = [
            p.parent
            for p in extracted.rglob("audio")
            if p.is_dir() and "__MACOSX" not in p.parts
        ]
        if len(candidates) != 1:
            raise ValueError("Expected one audio/ folder.")
        shutil.copytree(candidates[0], output)
    return output.resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--output", type=Path, default=ROOT_PATH / "data/demo_input")
    args = parser.parse_args()
    print(prepare_dataset(args.url, args.output))


if __name__ == "__main__":
    main()
