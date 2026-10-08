"""octree モジュールのテスト（locational code 対応）。"""

from __future__ import annotations

import logging

import numpy as np

from core.model import CellState, code_depth
from core.octree import build_voxel_model, make_cubic_bbox


def test_single_triangle_yields_shell_cells() -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([4.0, 4.0, 4.0])
    triangle = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]])
    facets = np.array([triangle])

    model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size=3.0)

    assert model.shell_count >= 1
    for cell in model.get_shell_cells():
        assert cell.state == CellState.SHELL
        assert cell.facet_indices is not None


def test_root_always_splits_at_least_once() -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([2.0, 2.0, 2.0])
    triangle = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]])
    facets = np.array([triangle])

    model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size=100.0)

    assert model.shell_count >= 1
    # ルート(depth 0)そのものではなく、必ず分割された子であること
    for cell in model.get_shell_cells():
        assert cell.depth >= 1


def test_smaller_voxel_size_yields_smaller_cells() -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([4.0, 4.0, 4.0])
    triangle = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]])
    facets = np.array([triangle])

    coarse = build_voxel_model(bbox_min, bbox_max, facets, voxel_size=3.0)
    fine = build_voxel_model(bbox_min, bbox_max, facets, voxel_size=1.5)

    coarse_size = coarse.cell_origin_size(coarse.get_shell_cells()[0])[1]
    fine_max_size = max(fine.cell_origin_size(c)[1].mean() for c in fine.get_shell_cells())
    assert fine_max_size < coarse_size.mean()


def test_max_depth_forces_cell_and_logs_warning(caplog) -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([8.0, 8.0, 8.0])
    triangle = np.array([[0.0, 0.0, 0.0], [8.0, 8.0, 0.0], [0.0, 8.0, 8.0]])
    facets = np.array([triangle])

    with caplog.at_level(logging.WARNING):
        model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size=1e-6, max_depth=2)

    assert model.shell_count > 0
    assert any("MAX_OCTREE_DEPTH" in record.message for record in caplog.records)
    # 強制確定されたセルは max_depth で止まっている
    for cell in model.get_shell_cells():
        assert cell.depth <= 2


def test_make_cubic_bbox_expands_to_longest_edge_centered() -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([10.0, 2.0, 4.0])

    cube_min, cube_max = make_cubic_bbox(bbox_min, bbox_max)

    assert np.allclose(cube_max - cube_min, [10.0, 10.0, 10.0])
    # 元のbboxの中心を保ったまま拡張されていること
    original_center = (bbox_min + bbox_max) / 2.0
    cube_center = (cube_min + cube_max) / 2.0
    assert np.allclose(cube_center, original_center)


def test_cubic_root_yields_cube_cells() -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([8.0, 2.0, 2.0])  # 非立方体のbbox
    triangle = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]])
    facets = np.array([triangle])

    model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size=3.0, cubic_root=True)

    assert model.shell_count >= 1
    for cell in model.get_shell_cells():
        _, size = model.cell_origin_size(cell)
        assert np.allclose(size[0], size[1])
        assert np.allclose(size[1], size[2])


def test_non_cubic_root_yields_non_cube_cells() -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([8.0, 2.0, 2.0])  # 非立方体のbbox
    triangle = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]])
    facets = np.array([triangle])

    model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size=1.0, cubic_root=False)

    assert model.shell_count >= 1
    # cubic_root無しだと非立方体セル（X辺が他より大きい）
    cell = model.get_shell_cells()[0]
    _, size = model.cell_origin_size(cell)
    assert size[0] > size[1]


def test_all_codes_have_valid_depth() -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([4.0, 4.0, 4.0])
    triangle = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]])
    facets = np.array([triangle])

    model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size=1.0)

    for code, cell in model.cells.items():
        # code から計算した depth と cell.depth が一致
        assert code_depth(code) == cell.depth
