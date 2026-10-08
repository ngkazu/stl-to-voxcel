"""shrink モジュール（面のマージ + 縮小）のテスト。"""

from __future__ import annotations

import numpy as np

from core.model import Cell, CellState, VoxelModel, code_from_grid_index
from core.shrink import greedy_merge_faces, shrink_model


def test_greedy_merge_contiguous_faces() -> None:
    model = VoxelModel(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        base_cell_size=np.array([4.0, 4.0, 4.0]),
    )
    depth = 1
    for i in (0, 1):
        for j in (0, 1):
            code = code_from_grid_index((i, j, 1), depth)
            model.add_cell(Cell(code=code, state=CellState.SHELL))

    merged = greedy_merge_faces(model, axis=2, direction=1)

    assert len(merged) == 1
    assert len(merged[0].source_codes) == 4
    assert merged[0].rect == (0.0, 4.0, 0.0, 4.0)


def test_greedy_merge_does_not_cross_different_sizes() -> None:
    model = VoxelModel(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        base_cell_size=np.array([4.0, 4.0, 4.0]),
    )
    code_d1 = code_from_grid_index((0, 0, 1), 1)
    model.add_cell(Cell(code=code_d1, state=CellState.SHELL))

    code_d2 = code_from_grid_index((1, 1, 3), 2)
    model.add_cell(Cell(code=code_d2, state=CellState.SHELL))

    merged = greedy_merge_faces(model, axis=2, direction=1)

    # 異なるdepth(サイズ)のセルはマージされず、別々のMergedFaceになる
    assert len(merged) == 2


def test_greedy_merge_facet_indices_union() -> None:
    model = VoxelModel(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        base_cell_size=np.array([4.0, 4.0, 4.0]),
    )
    depth = 1
    code_a = code_from_grid_index((0, 0, 1), depth)
    code_b = code_from_grid_index((1, 0, 1), depth)
    model.add_cell(Cell(code=code_a, state=CellState.SHELL, facet_indices=(0, 1)))
    model.add_cell(Cell(code=code_b, state=CellState.SHELL, facet_indices=(1, 2)))

    merged = greedy_merge_faces(model, axis=2, direction=1)

    assert len(merged) == 1
    assert merged[0].facet_indices == (0, 1, 2)


def test_shrink_single_face_only() -> None:
    model = VoxelModel(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        base_cell_size=np.array([8.0, 8.0, 8.0]),
    )
    depth = 2  # dim=4, セルサイズ = 8/4 = 2
    target = Cell(
        code=code_from_grid_index((1, 1, 1), depth), state=CellState.SHELL, facet_indices=(0,)
    )
    model.add_cell(target)

    # +X以外の5方向はINSIDEセルを登録し、外殻面（縮小対象）にならないようにする
    for axis, direction in [(0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
        idx = [1, 1, 1]
        idx[axis] += direction
        neighbor = code_from_grid_index((idx[0], idx[1], idx[2]), depth)
        model.add_cell(Cell(code=neighbor, state=CellState.INSIDE))

    # targetセルの範囲: origin=(2,2,2), size=(2,2,2) -> [2,4]x[2,4]x[2,4]
    # +X面(x=4)から0.5だけ内側(x=3.5)に平らな三角形を置く
    triangle = np.array([[3.5, 2.5, 2.5], [3.5, 3.5, 2.5], [3.5, 2.5, 3.5]])
    facets = np.array([triangle])

    shrink_model(model, facets)

    face_positions = model.cell_face_positions(target)
    assert np.isclose(face_positions[1], 3.5)  # +X面だけが縮む
    assert np.isclose(face_positions[0], 2.0)
    assert np.isclose(face_positions[2], 2.0)
    assert np.isclose(face_positions[3], 4.0)
    assert np.isclose(face_positions[4], 2.0)
    assert np.isclose(face_positions[5], 4.0)


def test_shrink_preserves_inside() -> None:
    model = VoxelModel(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        base_cell_size=np.array([4.0, 4.0, 4.0]),
    )
    depth = 1
    shell = Cell(
        code=code_from_grid_index((0, 0, 0), depth), state=CellState.SHELL, facet_indices=(0,)
    )
    inside = Cell(code=code_from_grid_index((1, 0, 0), depth), state=CellState.INSIDE)
    model.add_cell(shell)
    model.add_cell(inside)

    triangle = np.array([[1.5, 0.5, 0.5], [1.5, 1.5, 0.5], [1.5, 0.5, 1.5]])
    facets = np.array([triangle])

    shrink_model(model, facets)

    assert inside.face_positions is None  # INSIDEセルはshrinkの対象外
