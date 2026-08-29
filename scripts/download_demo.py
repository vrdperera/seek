"""Download and install the portable StyleSeek demonstration artifacts."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from styleseek.paths import PROJECT_ROOT


DEFAULT_URL = (
    "https://github.com/vrdperera/seek/releases/download/"
    "v1.0-demo/styleseek-demo-v1.zip"
)
REQUIRED_FILES = [
    "yolo11n.pt",
    "artifacts/checkpoints/detector/best.pt",
    "artifacts/checkpoints/retrieval/best.pt",
    "artifacts/indexes/catalogue.pt",
    "artifacts/metadata/release_manifest.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install StyleSeek demo artifacts")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--checksum-url")
    parser.add_argument("--destination", default=str(PROJECT_ROOT))
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "StyleSeek-setup/1.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_checksum(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip().split()
    if not value or len(value[0]) != 64:
        raise ValueError("Downloaded checksum file is invalid")
    return value[0].lower()


def safe_extract(archive_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(root)


def validate_installation(destination: Path) -> None:
    missing = [relative for relative in REQUIRED_FILES if not (destination / relative).is_file()]
    if missing:
        values = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"The installed bundle is incomplete:\n{values}")


def main() -> None:
    args = parse_args()
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checksum_url = args.checksum_url or f"{args.url}.sha256"

    with tempfile.TemporaryDirectory(prefix="styleseek-download-") as directory:
        temporary = Path(directory)
        archive = temporary / "styleseek-demo.zip"
        checksum_file = temporary / "styleseek-demo.zip.sha256"
        print(f"Downloading {args.url}")
        download(args.url, archive)
        download(checksum_url, checksum_file)
        expected = expected_checksum(checksum_file)
        actual = sha256(archive)
        if actual != expected:
            raise ValueError(
                f"Artifact checksum mismatch: expected {expected}, received {actual}"
            )
        safe_extract(archive, destination)

    validate_installation(destination)
    print(f"Installed StyleSeek demo artifacts in {destination}")
    print("Run the application with: uv run python app.py")


if __name__ == "__main__":
    main()
