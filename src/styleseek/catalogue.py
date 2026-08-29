"""Portable catalogue-index path handling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .paths import PROJECT_ROOT


def make_portable_paths(
    paths: Iterable[str | Path], portable_root: str | Path = PROJECT_ROOT
) -> tuple[list[str], str]:
    """Store project-contained paths relatively and retain external paths explicitly."""

    root = Path(portable_root).resolve()
    resolved = [Path(path).resolve() for path in paths]
    try:
        return [path.relative_to(root).as_posix() for path in resolved], "project_root"
    except ValueError:
        return [str(path) for path in resolved], "absolute"


def resolve_catalogue_paths(
    paths: Iterable[str | Path],
    path_base: str | None,
    project_root: str | Path = PROJECT_ROOT,
) -> list[str]:
    """Resolve portable paths after a project has been cloned or moved."""

    root = Path(project_root).resolve()
    values = [Path(path) for path in paths]
    if path_base == "project_root":
        return [str((root / path).resolve()) for path in values]
    if path_base in {None, "absolute"}:
        return [str(path.resolve()) for path in values]
    raise ValueError(f"Unsupported catalogue path base: {path_base!r}")


def require_catalogue_images(paths: Iterable[str | Path]) -> None:
    """Fail early with an actionable message when a release is incomplete."""

    values = [Path(path) for path in paths]
    missing = [path for path in values if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"Catalogue index references {len(missing)} missing image(s): {preview}. "
            "Run `uv run python scripts/download_demo.py` to install the demo artifacts."
        )
