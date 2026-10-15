"""VoxelModel（SHELL+INSIDE）をSTLとして書き出す。

`mode`で出力形式を切り替える:
- "all"（デフォルト）: 各セルの全6面を個別に出力する（内部面も含むフルボックス出力）。
  後段でBoolean Union等によりソリッド化することを前提に、1セル=1個のボックスとして書き出す。
- "boundary": OUTSIDE/EMPTY/未登録に接する境界面だけを（マージせず）出力する。
- "merged": 境界面を同一平面・同一サイズでgreedy meshマージしてから出力する
  （三角形数を削減できるが、shrink後は角・エッジ部で歯抜けが起きうる）。
shrink済みのモデルであれば、cell_face_positions経由で縮小後の実形状が反映される。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from stl import mesh as stl_mesh

from core.model import Cell, CellState, VoxelModel
from core.shrink import MergedFace, greedy_merge_faces

ExportMode = Literal["all", "boundary", "merged"]


def export_voxel_stl(model: VoxelModel, path: str, mode: ExportMode = "all") -> int:
    """SHELL+INSIDEセルの面をSTLとして書き出し、出力した三角形数を返す。"""
    cells = model.get_shell_cells() + model.get_inside_cells()

    triangles: list[np.ndarray] = []
    if mode == "merged":
        for axis in range(3):
            for direction in (1, -1):
                for merged in greedy_merge_faces(model, axis, direction, cells=cells):
                    triangles.extend(_merged_face_to_triangles(merged))
    else:
        boundary_only = mode == "boundary"
        for axis in range(3):
            for direction in (1, -1):
                for cell in cells:
                    if boundary_only:
                        neighbor_state = model.get_neighbor_state(cell, axis, direction)
                        if neighbor_state is not None and neighbor_state != CellState.OUTSIDE:
                            continue  # 占有セルに接している内部面は除外
                    triangles.extend(
                        _merged_face_to_triangles(_cell_face(model, cell, axis, direction))
                    )

    facets = np.array(triangles, dtype=np.float32)
    data = np.zeros(len(facets), dtype=stl_mesh.Mesh.dtype)
    data["vectors"] = facets
    stl_mesh.Mesh(data).save(path)
    return len(facets)


def _cell_face(model: VoxelModel, cell: Cell, axis: int, direction: int) -> MergedFace:
    """1セルの1面を（マージせず）MergedFace形式で返す。"""
    axes = [a for a in range(3) if a != axis]
    axis_a, axis_b = axes
    face_positions = model.cell_face_positions(cell)
    a_min, a_max = face_positions[axis_a * 2], face_positions[axis_a * 2 + 1]
    b_min, b_max = face_positions[axis_b * 2], face_positions[axis_b * 2 + 1]
    axis_min, axis_max = face_positions[axis * 2], face_positions[axis * 2 + 1]
    return MergedFace(
        axis=axis,
        direction=direction,
        rect=(a_min, a_max, b_min, b_max),
        face_coord=axis_max if direction > 0 else axis_min,
        cell_size_axis=axis_max - axis_min,
        source_codes=(cell.code,),
        facet_indices=cell.facet_indices or (),
    )


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
