"""export モジュール（VoxelModelのSTL書き出し）のテスト。"""

from __future__ import annotations

import numpy as np
from stl import mesh as stl_mesh

from core.export import export_voxel_stl
from core.model import Cell, CellState, VoxelModel, code_from_grid_index


def test_export_default_single_cell_is_twelve_triangles(tmp_path) -> None:
    model = VoxelModel(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        base_cell_size=np.array([2.0, 2.0, 2.0]),
    )
    depth = 1
    model.add_cell(Cell(code=code_from_grid_index((0, 0, 0), depth), state=CellState.SHELL))

    out_path = tmp_path / "single.stl"
    n_triangles = export_voxel_stl(model, str(out_path))

    assert n_triangles == 12
    loaded = stl_mesh.Mesh.from_file(str(out_path))
    assert len(loaded.vectors) == 12


def test_export_default_includes_internal_face_between_adjacent_cells(tmp_path) -> None:
    model = VoxelModel(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        base_cell_size=np.array([2.0, 2.0, 2.0]),
    )
    depth = 1
    # X方向に隣接する2セル。デフォルトは各セルの全6面を個別出力するため、
    # 内部面(x=1)も両セル分(2枚)含めて出力される: 2セル x 6面 x 2三角形 = 24
    model.add_cell(Cell(code=code_from_grid_index((0, 0, 0), depth), state=CellState.SHELL))
    model.add_cell(Cell(code=code_from_grid_index((1, 0, 0), depth), state=CellState.SHELL))

    out_path = tmp_path / "pair.stl"
    n_triangles = export_voxel_stl(model, str(out_path))

    assert n_triangles == 24


def test_export_boundary_only_excludes_internal_face(tmp_path) -> None:
    model = VoxelModel(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        base_cell_size=np.array([2.0, 2.0, 2.0]),
    )
    depth = 1
    model.add_cell(Cell(code=code_from_grid_index((0, 0, 0), depth), state=CellState.SHELL))
    model.add_cell(Cell(code=code_from_grid_index((1, 0, 0), depth), state=CellState.SHELL))

    out_path = tmp_path / "pair_boundary_only.stl"
    n_triangles = export_voxel_stl(model, str(out_path), boundary_only=True)

    # 内部面(x=1)は除外される。外側は(+X:1面 -X:1面 +-Y/+-Z:各2面)=10面 x 2三角形 = 20
    assert n_triangles == 20


def test_export_boundary_only_treats_inside_cells_like_shell(tmp_path) -> None:
    model = VoxelModel(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        base_cell_size=np.array([2.0, 2.0, 2.0]),
    )
    depth = 1
    model.add_cell(Cell(code=code_from_grid_index((0, 0, 0), depth), state=CellState.INSIDE))
    model.add_cell(Cell(code=code_from_grid_index((1, 0, 0), depth), state=CellState.SHELL))

    out_path = tmp_path / "mixed_boundary_only.stl"
    n_triangles = export_voxel_stl(model, str(out_path), boundary_only=True)

    # SHELL/INSIDEの区別なく「占有セル」として扱われ、内部面が除外される
    assert n_triangles == 20


def test_export_full_block_default_includes_all_internal_faces(tmp_path) -> None:
    model = VoxelModel(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        base_cell_size=np.array([2.0, 2.0, 2.0]),
    )
    depth = 1
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                model.add_cell(
                    Cell(code=code_from_grid_index((i, j, k), depth), state=CellState.SHELL)
                )

    out_path = tmp_path / "block.stl"
    n_triangles = export_voxel_stl(model, str(out_path))

    # 8セル x 6面 x 2三角形 = 96（内部面も含めてすべて出力）
    assert n_triangles == 96


def test_export_full_block_boundary_only(tmp_path) -> None:
    model = VoxelModel(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        base_cell_size=np.array([2.0, 2.0, 2.0]),
    )
    depth = 1
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                model.add_cell(
                    Cell(code=code_from_grid_index((i, j, k), depth), state=CellState.SHELL)
                )

    out_path = tmp_path / "block_boundary_only.stl"
    n_triangles = export_voxel_stl(model, str(out_path), boundary_only=True)

    # 2x2x2の各セルは軸ごとに必ず片方の面だけが外側に露出する(1セルあたり3面)
    # 8セル x 3面 x 2三角形 = 48（マージはしない）
    assert n_triangles == 48
