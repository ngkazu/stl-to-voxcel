"""export モジュール（VoxelModelのSTL書き出し）のテスト。"""

from __future__ import annotations

import numpy as np
from stl import mesh as stl_mesh

from core.export import export_voxel_stl
from core.model import Cell, CellState, VoxelModel, code_from_grid_index


def test_export_single_cell_cube_is_twelve_triangles(tmp_path) -> None:
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


def test_export_merges_internal_face_between_adjacent_cells(tmp_path) -> None:
    model = VoxelModel(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        base_cell_size=np.array([2.0, 2.0, 2.0]),
    )
    depth = 1
    # X方向に隣接する2セル。内部面(x=1)は出力されず、外側の4面もgreedy mergeで
    # それぞれ1枚の矩形にまとまるため、合計は単一セルと同じ12三角形になるはず。
    model.add_cell(Cell(code=code_from_grid_index((0, 0, 0), depth), state=CellState.SHELL))
    model.add_cell(Cell(code=code_from_grid_index((1, 0, 0), depth), state=CellState.SHELL))

    out_path = tmp_path / "pair.stl"
    n_triangles = export_voxel_stl(model, str(out_path))

    assert n_triangles == 12


def test_export_no_merge_outputs_face_per_cell(tmp_path) -> None:
    model = VoxelModel(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        base_cell_size=np.array([2.0, 2.0, 2.0]),
    )
    depth = 1
    model.add_cell(Cell(code=code_from_grid_index((0, 0, 0), depth), state=CellState.SHELL))
    model.add_cell(Cell(code=code_from_grid_index((1, 0, 0), depth), state=CellState.SHELL))

    out_path = tmp_path / "pair_no_merge.stl"
    n_triangles = export_voxel_stl(model, str(out_path), merge_faces=False)

    # 内部面(x=1)は除外されるが、外側4面はセルごとに個別出力されるためマージ時より増える
    # (+X:1面 -X:1面 +-Y/+-Z:各2面) = 2 + 4*2 = 10面 x 2三角形 = 20
    assert n_triangles == 20


def test_export_treats_inside_cells_like_shell(tmp_path) -> None:
    model = VoxelModel(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        base_cell_size=np.array([2.0, 2.0, 2.0]),
    )
    depth = 1
    model.add_cell(Cell(code=code_from_grid_index((0, 0, 0), depth), state=CellState.INSIDE))
    model.add_cell(Cell(code=code_from_grid_index((1, 0, 0), depth), state=CellState.SHELL))

    out_path = tmp_path / "mixed.stl"
    n_triangles = export_voxel_stl(model, str(out_path))

    # SHELL/INSIDEの区別なく「占有セル」として扱われ、内部面が除外されマージされる
    assert n_triangles == 12


def test_export_full_block_merges_to_single_box(tmp_path) -> None:
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

    # 2x2x2セルがすべて埋まった立方体は、外側境界だけ見れば1つの大きな箱と同じ
    assert n_triangles == 12
