import json
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from src.utils.io_utils import ROOT_PATH
import gdown


def download(url, output):
    if urlparse(url).hostname == "drive.google.com":
        if gdown.download(url, str(output), fuzzy=True, quiet=False) is None:
            raise RuntimeError("Download failed.")
    else:
        with urlopen(url, timeout=120) as source, Path(output).open("wb") as destination:
            shutil.copyfileobj(source, destination)


def download_resources():
    resources = json.loads((ROOT_PATH / "resources.json").read_text())
    for item in resources["files"]:
        destination = ROOT_PATH / item["path"]
        if destination.is_file():
            print("Already downloaded:", item["path"])
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            downloaded = Path(temporary) / "download"
            download(item["url"], downloaded)
            downloaded.replace(destination)
        print("Downloaded:", item["path"])


if __name__ == "__main__":
    download_resources()
