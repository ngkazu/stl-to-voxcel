"""numpy-stl を使ったSTL読み込みとバウンディングボックス取得。"""

from __future__ import annotations

import numpy as np
from stl import mesh

_AREA_EPS = 1e-12


def load_stl(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STLファイルを読み込み、(facets, bbox_min, bbox_max) を返す。

    facets は面積がほぼゼロの退化ファセットを除外した (N, 3, 3) の float64 配列。
    """
    stl_mesh = mesh.Mesh.from_file(path)
    facets = _remove_degenerate_facets(stl_mesh.vectors.astype(np.float64))

    points = facets.reshape(-1, 3)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    return facets, bbox_min, bbox_max


def _remove_degenerate_facets(facets: np.ndarray) -> np.ndarray:
    """面積がほぼゼロのファセットを除外する。"""
    edge0 = facets[:, 1] - facets[:, 0]
    edge1 = facets[:, 2] - facets[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge0, edge1), axis=1)
    return facets[areas > _AREA_EPS]
