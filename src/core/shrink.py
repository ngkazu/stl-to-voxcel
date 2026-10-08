"""Shrink（縮小）機能: SHELLセルの外殻面をSTL表面まで縮める。

8分木のセル構造（code）はそのまま維持し、各セルの face_positions だけを更新する。
同一サイズ・同一平面上で隣接する外殻面を greedy meshing でマージしてから、面積ベースの
スイープ距離計算（core.geometry.sweep_distance_to_facets）でまとめて移動量を求める。

設計の詳細は memos/shrink_design.md を参照。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.geometry import sweep_distance_to_facets
from core.model import Cell, CellState, VoxelModel, code_to_grid_index


@dataclass
class MergedFace:
    """greedy meshingでマージされた外殻面（1枚の矩形）。"""

    axis: int  # スイープする軸 (0=X, 1=Y, 2=Z)
    direction: int  # 面の向き（+1=axis正方向側の面, -1=axis負方向側の面）
    rect: tuple[float, float, float, float]  # スイープしない2軸の範囲 (min_a,max_a,min_b,max_b)
    face_coord: float  # スイープ軸上の面の現在位置（絶対座標）
    cell_size_axis: float  # マージ元セル（同一サイズ）のaxis方向の辺長
    source_codes: tuple[int, ...]  # マージ元セルのcode
    facet_indices: tuple[int, ...]  # マージ元セルのfacet_indicesの和集合


def shrink_model(model: VoxelModel, facets: np.ndarray) -> None:
    """全SHELLセルの外殻面を、STL表面まで縮める（face_positionsをin-placeで更新する）。"""
    facets = np.asarray(facets, dtype=np.float64)
    for axis in range(3):
        for direction in (1, -1):
            for merged in greedy_merge_faces(model, axis, direction):
                _shrink_merged_face(model, merged, facets)


def _shrink_merged_face(model: VoxelModel, merged: MergedFace, facets: np.ndarray) -> None:
    if not merged.facet_indices:
        return

    candidates = facets[list(merged.facet_indices)]
    distance = sweep_distance_to_facets(
        face_coord=merged.face_coord,
        rect=merged.rect,
        axis=merged.axis,
        direction=-merged.direction,  # 面の法線の逆方向（外側→内側）へ探索する
        facets=candidates,
    )
    if distance is None or distance >= merged.cell_size_axis:
        return  # 候補が無い、またはこのセルの深さを超える位置なら移動しない

    face_index = merged.axis * 2 + (1 if merged.direction > 0 else 0)
    for code in merged.source_codes:
        cell = model.cells[code]
        face_positions = model.cell_face_positions(cell).copy()
        face_positions[face_index] += distance * (-merged.direction)
        cell.face_positions = face_positions


def greedy_merge_faces(
    model: VoxelModel,
    axis: int,
    direction: int,
    cells: list[Cell] | None = None,
) -> list[MergedFace]:
    """同一サイズ・同一平面上で隣接する外殻面を greedy meshing でマージする。

    対象は、指定方向の隣接セルが OUTSIDE または存在しない（境界外/EMPTY）セル。
    マージは同一depth（同一サイズ）・同一レイヤー（スイープ軸のグリッドインデックスが同じ）
    のセル同士でのみ行う。

    Args:
        cells: マージ対象の候補セル集合。Noneなら `model.get_shell_cells()`
            （shrinkでの従来の挙動と後方互換）。STL出力等でSHELL+INSIDEを対象にする場合は
            呼び出し側から明示的に渡す。
    """
    axes = [a for a in range(3) if a != axis]
    axis_a, axis_b = axes

    target_cells = model.get_shell_cells() if cells is None else cells

    # (depth, スイープ軸グリッドインデックス) ごとに、2Dグリッド(axis_a,axis_b) -> Cell を集める
    by_depth_layer: dict[tuple[int, int], dict[tuple[int, int], Cell]] = {}
    for cell in target_cells:
        if model.get_neighbor_state(cell, axis, direction) not in (None, CellState.OUTSIDE):
            continue
        index = code_to_grid_index(cell.code)
        key = (cell.depth, index[axis])
        by_depth_layer.setdefault(key, {})[(index[axis_a], index[axis_b])] = cell

    merged_faces: list[MergedFace] = []
    for index_map in by_depth_layer.values():
        merged_faces.extend(_greedy_merge_layer(model, index_map, axis, axis_a, axis_b, direction))
    return merged_faces


def _row_fully_available(
    index_map: dict[tuple[int, int], Cell],
    visited: set[tuple[int, int]],
    a0: int,
    b: int,
    width: int,
) -> bool:
    return all((a0 + d, b) in index_map and (a0 + d, b) not in visited for d in range(width))


def _greedy_merge_layer(
    model: VoxelModel,
    index_map: dict[tuple[int, int], Cell],
    axis: int,
    axis_a: int,
    axis_b: int,
    direction: int,
) -> list[MergedFace]:
    """1つの(depth, layer)内の2Dグリッド上で、連続する矩形領域をマージする（greedy meshing）。"""
    visited: set[tuple[int, int]] = set()
    merged_faces: list[MergedFace] = []

    for a0, b0 in sorted(index_map):
        if (a0, b0) in visited:
            continue

        width = 1
        while (a0 + width, b0) in index_map and (a0 + width, b0) not in visited:
            width += 1

        height = 1
        while _row_fully_available(index_map, visited, a0, b0 + height, width):
            height += 1

        source_cells: list[Cell] = []
        for da in range(width):
            for db in range(height):
                key = (a0 + da, b0 + db)
                source_cells.append(index_map[key])
                visited.add(key)

        merged_faces.append(
            _build_merged_face(model, source_cells, axis, axis_a, axis_b, direction)
        )

    return merged_faces


def _build_merged_face(
    model: VoxelModel,
    source_cells: list[Cell],
    axis: int,
    axis_a: int,
    axis_b: int,
    direction: int,
) -> MergedFace:
    min_a = min_b = float("inf")
    max_a = max_b = float("-inf")
    facet_indices: set[int] = set()
    face_coord = 0.0
    cell_size_axis = 0.0

    for cell in source_cells:
        # cell_face_positions() はshrink後の実位置（未shrinkなら元のorigin/size由来）を返す
        face_positions = model.cell_face_positions(cell)
        a_min, a_max = face_positions[axis_a * 2], face_positions[axis_a * 2 + 1]
        b_min, b_max = face_positions[axis_b * 2], face_positions[axis_b * 2 + 1]
        min_a = min(min_a, a_min)
        max_a = max(max_a, a_max)
        min_b = min(min_b, b_min)
        max_b = max(max_b, b_max)
        if cell.facet_indices:
            facet_indices.update(cell.facet_indices)
        axis_min, axis_max = face_positions[axis * 2], face_positions[axis * 2 + 1]
        face_coord = axis_max if direction > 0 else axis_min
        cell_size_axis = axis_max - axis_min

    return MergedFace(
        axis=axis,
        direction=direction,
        rect=(min_a, max_a, min_b, max_b),
        face_coord=face_coord,
        cell_size_axis=cell_size_axis,
        source_codes=tuple(c.code for c in source_cells),
        facet_indices=tuple(sorted(facet_indices)),
    )
