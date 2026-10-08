"""VoxelModel（SHELL+INSIDE）をSTLとして書き出す。

shrinkで使うgreedy meshing（面のマージ、core.shrink.greedy_merge_faces）を再利用し、
占有セル同士の内部面を除いた境界面（OUTSIDE/EMPTYと接する面）だけを三角形化して
STL出力する。shrink済みのモデルであれば、cell_face_positions経由で縮小後の実形状が
反映される。
"""

from __future__ import annotations

import numpy as np
from stl import mesh as stl_mesh

from core.model import Cell, CellState, VoxelModel
from core.shrink import MergedFace, greedy_merge_faces


def export_voxel_stl(model: VoxelModel, path: str, merge_faces: bool = True) -> int:
    """SHELL+INSIDEセルの境界面だけをSTLとして書き出し、出力した三角形数を返す。

    merge_faces=False にすると、greedy mergingを行わずセル単位で面を出力する
    （歯抜け原因の切り分け用）。
    """
    cells = model.get_shell_cells() + model.get_inside_cells()

    triangles: list[np.ndarray] = []
    for axis in range(3):
        for direction in (1, -1):
            if merge_faces:
                merged_faces = greedy_merge_faces(model, axis, direction, cells=cells)
            else:
                merged_faces = _unmerged_faces(model, cells, axis, direction)
            for merged in merged_faces:
                triangles.extend(_merged_face_to_triangles(merged))

    facets = np.array(triangles, dtype=np.float32)
    data = np.zeros(len(facets), dtype=stl_mesh.Mesh.dtype)
    data["vectors"] = facets
    stl_mesh.Mesh(data).save(path)
    return len(facets)


def _unmerged_faces(
    model: VoxelModel, cells: list[Cell], axis: int, direction: int
) -> list[MergedFace]:
    """セルごとに、マージせず境界面を1枚ずつ作る。"""
    axes = [a for a in range(3) if a != axis]
    axis_a, axis_b = axes

    faces = []
    for cell in cells:
        neighbor_state = model.get_neighbor_state(cell, axis, direction)
        if neighbor_state is not None and neighbor_state != CellState.OUTSIDE:
            continue  # 占有セルに接している内部面は除外

        face_positions = model.cell_face_positions(cell)
        a_min, a_max = face_positions[axis_a * 2], face_positions[axis_a * 2 + 1]
        b_min, b_max = face_positions[axis_b * 2], face_positions[axis_b * 2 + 1]
        axis_min, axis_max = face_positions[axis * 2], face_positions[axis * 2 + 1]
        faces.append(
            MergedFace(
                axis=axis,
                direction=direction,
                rect=(a_min, a_max, b_min, b_max),
                face_coord=axis_max if direction > 0 else axis_min,
                cell_size_axis=axis_max - axis_min,
                source_codes=(cell.code,),
                facet_indices=cell.facet_indices or (),
            )
        )
    return faces


def _merged_face_to_triangles(face: MergedFace) -> list[np.ndarray]:
    """1枚のMergedFaceを、外向き法線が正しくなるよう2枚の三角形に変換する。"""
    axis = face.axis
    axes = [a for a in range(3) if a != axis]
    axis_a, axis_b = axes
    min_a, max_a, min_b, max_b = face.rect

    def point(a_val: float, b_val: float) -> np.ndarray:
        p = np.zeros(3)
        p[axis] = face.face_coord
        p[axis_a] = a_val
        p[axis_b] = b_val
        return p

    p00 = point(min_a, min_b)
    p10 = point(max_a, min_b)
    p11 = point(max_a, max_b)
    p01 = point(min_a, max_b)

    desired_normal = np.zeros(3)
    desired_normal[axis] = float(face.direction)

    def oriented(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        # 計算した法線が外向き(desired_normal)と逆なら頂点順を入れ替える
        normal = np.cross(v1 - v0, v2 - v0)
        if np.dot(normal, desired_normal) < 0:
            return np.array([v0, v2, v1])
        return np.array([v0, v1, v2])

    return [oriented(p00, p10, p11), oriented(p00, p11, p01)]
