"""Скачивает русскую часть датасета MASSIVE 1.1 в data/raw/massive/.

uv run python scripts/download_massive.py
"""

import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
MEMBERS = {"1.1/data/ru-RU.jsonl": "ru-RU.jsonl", "1.1/LICENSE": "LICENSE"}
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "massive"


def download(url: str, path: Path) -> None:
    with urllib.request.urlopen(url) as response, path.open("wb") as out:
        shutil.copyfileobj(response, out)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if all((OUT_DIR / name).exists() for name in MEMBERS.values()):
        print(f"уже скачано: {OUT_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "massive.tar.gz"
        print(f"скачиваю {URL}")
        download(URL, archive)
        if sha256(archive) != SHA256:
            raise SystemExit("контрольная сумма архива не совпала")

        with tarfile.open(archive) as tar:
            for member, name in MEMBERS.items():
                source = tar.extractfile(member)
                if source is None:
                    raise SystemExit(f"в архиве нет {member}")
                with source, (OUT_DIR / name).open("wb") as out:
                    shutil.copyfileobj(source, out)

    print(f"готово: {OUT_DIR}")


if __name__ == "__main__":
    main()
