from __future__ import annotations

from pathlib import Path

import numpy as np
from stl import mesh as stl_mesh

from core.stl_loader import load_stl


def _save_mesh(path: Path, vectors: np.ndarray) -> None:
    data = np.zeros(vectors.shape[0], dtype=stl_mesh.Mesh.dtype)
    data["vectors"] = vectors
    stl_mesh.Mesh(data, remove_empty_areas=False).save(str(path))


def test_load_stl_returns_facets_and_bbox(tmp_path: Path) -> None:
    vectors = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=np.float32,
    )
    stl_path = tmp_path / "triangle.stl"
    _save_mesh(stl_path, vectors)

    facets, bbox_min, bbox_max = load_stl(str(stl_path))

    assert facets.shape == (1, 3, 3)
    assert np.allclose(bbox_min, [0.0, 0.0, 0.0])
    assert np.allclose(bbox_max, [1.0, 1.0, 0.0])


def test_load_stl_filters_degenerate_facets(tmp_path: Path) -> None:
    vectors = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # 面積あり
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # 面積ゼロ(退化)
        ],
        dtype=np.float32,
    )
    stl_path = tmp_path / "with_degenerate.stl"
    _save_mesh(stl_path, vectors)

    facets, _bbox_min, _bbox_max = load_stl(str(stl_path))

    assert facets.shape == (1, 3, 3)
