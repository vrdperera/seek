from pathlib import Path

import pytest

from styleseek.catalogue import (
    make_portable_paths,
    require_catalogue_images,
    resolve_catalogue_paths,
)


def test_project_catalogue_paths_survive_moving_to_another_computer(tmp_path):
    source_root = tmp_path / "source"
    image = source_root / "demo_catalogue" / "images" / "product.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")

    stored, path_base = make_portable_paths([image], source_root)
    destination_root = tmp_path / "clone"
    expected = destination_root / "demo_catalogue" / "images" / "product.jpg"

    assert stored == ["demo_catalogue/images/product.jpg"]
    assert path_base == "project_root"
    assert resolve_catalogue_paths(stored, path_base, destination_root) == [
        str(expected.resolve())
    ]


def test_external_catalogue_paths_remain_absolute(tmp_path):
    project_root = tmp_path / "project"
    external = tmp_path / "external" / "product.jpg"

    stored, path_base = make_portable_paths([external], project_root)

    assert path_base == "absolute"
    assert stored == [str(external.resolve())]


def test_missing_catalogue_images_produce_setup_instruction(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/download_demo.py"):
        require_catalogue_images([tmp_path / "missing.jpg"])
