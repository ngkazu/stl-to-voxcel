"""solid_fill モジュールのテスト（locational code 対応）。

合成した直方体STLを使い、内部充填（INSIDEセル生成）を検証する。
"""

from __future__ import annotations

import numpy as np

from core.model import CellState
from core.octree import build_voxel_model
from core.solid_fill import fill_solid


def _box_facets(sx: float, sy: float, sz: float) -> np.ndarray:
    """原点を一端とする直方体 (sx, sy, sz) の三角形メッシュ (12, 3, 3) を返す。"""
    v = np.array(
        [
            [0, 0, 0],
            [sx, 0, 0],
            [sx, sy, 0],
            [0, sy, 0],
            [0, 0, sz],
            [sx, 0, sz],
            [sx, sy, sz],
            [0, sy, sz],
        ],
        dtype=np.float64,
    )
    faces = [
        [0, 1, 2],
        [0, 2, 3],  # bottom
        [4, 6, 5],
        [4, 7, 6],  # top
        [0, 4, 5],
        [0, 5, 1],  # front
        [1, 5, 6],
        [1, 6, 2],  # right
        [2, 6, 7],
        [2, 7, 3],  # back
        [3, 7, 4],
        [3, 4, 0],  # left
    ]
    return np.array([[v[i], v[j], v[k]] for i, j, k in faces])


def test_fill_solid_cube_fills_interior() -> None:
    """中実の立方体は内部がINSIDEで充填される。"""
    facets = _box_facets(10.0, 10.0, 10.0)
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([10.0, 10.0, 10.0])

    model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size=2.0, cubic_root=True)
    assert model.shell_count > 0

    cell_size = model.get_min_cell_size()
    fill_solid(model, cell_size)

    # 内部が充填される
    assert model.inside_count > 0


def test_fill_solid_preserves_shell() -> None:
    """充填処理でSHELLセル数は変わらない。"""
    facets = _box_facets(10.0, 10.0, 10.0)
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([10.0, 10.0, 10.0])

    model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size=2.0, cubic_root=True)
    shell_before = model.shell_count

    fill_solid(model, model.get_min_cell_size())

    assert model.shell_count == shell_before


def test_fill_solid_cell_states_consistent() -> None:
    """充填後の全セルが SHELL か INSIDE のいずれか。"""
    facets = _box_facets(10.0, 10.0, 10.0)
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([10.0, 10.0, 10.0])

    model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size=2.0, cubic_root=True)
    fill_solid(model, model.get_min_cell_size())

    for cell in model.cells.values():
        assert cell.state in (CellState.SHELL, CellState.INSIDE)

    assert model.cell_count == model.shell_count + model.inside_count


def test_fill_solid_inside_cells_within_bbox() -> None:
    """INSIDEセルは全てbbox内に存在する。"""
    facets = _box_facets(10.0, 10.0, 10.0)
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([10.0, 10.0, 10.0])

    model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size=2.0, cubic_root=True)
    fill_solid(model, model.get_min_cell_size())

    for cell in model.get_inside_cells():
        origin, size = model.cell_origin_size(cell)
        # bbox内（多少の許容）
        assert np.all(origin >= model.bbox_min - 1e-6)
        assert np.all(origin + size <= model.bbox_min + model.base_cell_size + 1e-6)


def test_fill_solid_non_cubic() -> None:
    """非立方体bbox（cubic_root無し）でも充填できる。"""
    facets = _box_facets(10.0, 6.0, 4.0)
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([10.0, 6.0, 4.0])

    model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size=1.0, cubic_root=False)
    fill_solid(model, model.get_min_cell_size())

    assert model.inside_count > 0
