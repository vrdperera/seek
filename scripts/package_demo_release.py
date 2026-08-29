"""Build a portable GitHub Release bundle for the StyleSeek demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from styleseek.index import build_catalogue_index
from styleseek.paths import DETECTOR_CHECKPOINT, PROJECT_ROOT, RETRIEVAL_CHECKPOINT


DEFAULT_OUTPUT = PROJECT_ROOT / "dist" / "styleseek-demo-v1.zip"
PERSON_CHECKPOINT = PROJECT_ROOT / "yolo11n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package trained weights and a generated portable demo catalogue"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--retrieval", default=str(RETRIEVAL_CHECKPOINT))
    parser.add_argument("--detector", default=str(DETECTOR_CHECKPOINT))
    parser.add_argument("--person-detector", default=str(PERSON_CHECKPOINT))
    parser.add_argument("--products", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--version", default="v1.0-demo")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_files(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        values = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required runtime artifacts are missing:\n{values}")


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_release_manifest(bundle_root: Path, version: str) -> None:
    files = {}
    for path in sorted(bundle_root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(bundle_root).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    payload = {
        "name": "StyleSeek AI portable demonstration bundle",
        "version": version,
        "catalogue": "Generated synthetic garments for software demonstration only",
        "files": files,
    }
    metadata = bundle_root / "artifacts" / "metadata" / "release_manifest.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_bundle_readme(bundle_root: Path) -> None:
    (bundle_root / "DEMO_BUNDLE_README.txt").write_text(
        "StyleSeek AI demonstration bundle\n"
        "=================================\n\n"
        "This bundle contains the trained detector and retrieval checkpoints, "
        "the COCO person-detector weights, and a small generated catalogue.\n\n"
        "The generated catalogue makes the application portable and does not "
        "replace the DeepFashion2 test set used for the metrics in the report.\n\n"
        "Extract this archive into the root of the StyleSeek repository, then run:\n"
        "  uv sync --locked\n"
        "  uv run python app.py\n",
        encoding="utf-8",
    )


def create_archive(bundle_root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(bundle_root).as_posix())


def main() -> None:
    args = parse_args()
    retrieval = Path(args.retrieval).resolve()
    detector = Path(args.detector).resolve()
    person_detector = Path(args.person_detector).resolve()
    require_files([retrieval, detector, person_detector])
    output = Path(args.output).resolve()

    with tempfile.TemporaryDirectory(prefix="styleseek-release-") as directory:
        bundle_root = Path(directory) / "bundle"
        copy_file(
            retrieval,
            bundle_root / "artifacts" / "checkpoints" / "retrieval" / "best.pt",
        )
        copy_file(
            detector,
            bundle_root / "artifacts" / "checkpoints" / "detector" / "best.pt",
        )
        copy_file(person_detector, bundle_root / "yolo11n.pt")

        sample_root = bundle_root / "data" / "samples"
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "generate_demo_data.py"),
                "--output",
                str(sample_root),
                "--products",
                str(args.products),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        build_catalogue_index(
            sample_root / "manifest.csv",
            bundle_root
            / "artifacts"
            / "checkpoints"
            / "retrieval"
            / "best.pt",
            bundle_root / "artifacts" / "indexes" / "catalogue.pt",
            split="test",
            device_name=args.device,
            portable_root=bundle_root,
        )
        write_bundle_readme(bundle_root)
        write_release_manifest(bundle_root, args.version)
        create_archive(bundle_root, output)

    checksum = sha256(output)
    checksum_path = Path(f"{output}.sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    print(f"Created {output} ({output.stat().st_size / (1024 * 1024):.1f} MiB)")
    print(f"Created {checksum_path}")


if __name__ == "__main__":
    main()
