"""Voxel群の膨張処理。

STL表面からユーザー指定の距離以上離れていることを保証するための処理を行う。
将来的には収縮（STL表面への密着）機能の追加も想定している。

アルゴリズム:
1. 占有セル（SHELL/INSIDE）に隣接する外部セルを8分木で生成
2. 外部セルのdepthが占有セルと異なる場合、最小Voxelサイズまで分割
3. 外部セルの頂点の距離を検証し、条件を満たすものだけOUTSIDEとして追加
4. 必要に応じて繰り返す
"""

from __future__ import annotations

import numpy as np

from core.geometry import MeshDistanceCalculator
from core.model import Cell, CellState, VoxelModel

# 6方向の隣接オフセット（単位ベクトル）
_NEIGHBOR_DIRECTIONS = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=np.float64,
)


def expand_model(
    model: VoxelModel,
    facets: np.ndarray,
    expand_distance: float,
    min_cell_size: float,
    max_iterations: int = 100,
) -> None:
    """VoxelModelを膨張させ、OUTSIDEセルを追加する。

    占有セル（SHELL/INSIDE）に隣接する外部空間を8分木で生成し、
    距離検証を通過したものだけOUTSIDEとして追加する。

    Args:
        model: VoxelModel（SHELL/INSIDEセルが登録済み）
        facets: 元STLの三角形群 (N, 3, 3)
        expand_distance: STL表面から離す距離
        min_cell_size: 最小セルサイズ（8分木分割の下限）
        max_iterations: 最大反復回数（無限ループ防止）
    """
    if expand_distance <= 0:
        return

    occupied_cells = model.get_occupied_cells()
    if not occupied_cells:
        return

    # KD-treeを使った距離計算器を準備
    dist_calc = MeshDistanceCalculator(facets)

    # 占有セルの空間インデックスを構築（高速な隣接判定用）
    # グリッドサイズはmin_cell_sizeを使用
    occupied_index, grid_size = _build_spatial_index(occupied_cells, min_cell_size)

    # 反復的に膨張処理
    for iteration in range(max_iterations):
        # 現在の占有セルに隣接する外部セル候補を生成
        candidates = _generate_adjacent_external_cells(
            model, occupied_index, grid_size, min_cell_size
        )

        if not candidates:
            break

        # 距離検証
        validated = _validate_cells_by_distance(candidates, dist_calc, expand_distance)

        if not validated:
            break

        # 検証済みセルをOUTSIDEとして追加
        for cell in validated:
            cell.state = CellState.OUTSIDE
            model.add_cell(cell)
            # 空間インデックスに追加
            _add_to_spatial_index(occupied_index, cell, grid_size)

    return


def _build_spatial_index(
    cells: list[Cell],
    grid_size: float,
) -> tuple[dict[tuple[int, int, int], list[Cell]], float]:
    """セルの空間インデックスを構築する。

    グリッドベースのインデックス。各グリッドセルに、その領域と重なるセルのリストを格納。

    Returns:
        (グリッドインデックス, グリッドサイズ)
    """
    index: dict[tuple[int, int, int], list[Cell]] = {}

    for cell in cells:
        # セルがカバーするグリッドセルを計算
        min_grid = (
            int(np.floor(cell.origin[0] / grid_size)),
            int(np.floor(cell.origin[1] / grid_size)),
            int(np.floor(cell.origin[2] / grid_size)),
        )
        max_grid = (
            int(np.floor((cell.origin[0] + cell.size) / grid_size)),
            int(np.floor((cell.origin[1] + cell.size) / grid_size)),
            int(np.floor((cell.origin[2] + cell.size) / grid_size)),
        )

        for gx in range(min_grid[0], max_grid[0] + 1):
            for gy in range(min_grid[1], max_grid[1] + 1):
                for gz in range(min_grid[2], max_grid[2] + 1):
                    key = (gx, gy, gz)
                    if key not in index:
                        index[key] = []
                    index[key].append(cell)

    return index, grid_size


def _add_to_spatial_index(
    index: dict[tuple[int, int, int], list[Cell]],
    cell: Cell,
    grid_size: float,
) -> None:
    """セルを空間インデックスに追加する。"""
    min_grid = (
        int(np.floor(cell.origin[0] / grid_size)),
        int(np.floor(cell.origin[1] / grid_size)),
        int(np.floor(cell.origin[2] / grid_size)),
    )
    max_grid = (
        int(np.floor((cell.origin[0] + cell.size) / grid_size)),
        int(np.floor((cell.origin[1] + cell.size) / grid_size)),
        int(np.floor((cell.origin[2] + cell.size) / grid_size)),
    )

    for gx in range(min_grid[0], max_grid[0] + 1):
        for gy in range(min_grid[1], max_grid[1] + 1):
            for gz in range(min_grid[2], max_grid[2] + 1):
                key = (gx, gy, gz)
                if key not in index:
                    index[key] = []
                index[key].append(cell)


def _is_position_overlapping_any(
    index: dict[tuple[int, int, int], list[Cell]],
    grid_size: float,
    origin: np.ndarray,
    size: float,
    tolerance: float = 1e-9,
) -> bool:
    """指定した位置が既存の占有セルと重なるかチェックする。

    グリッドインデックスを使って高速に判定する。
    """
    candidate_min = origin
    candidate_max = origin + size

    # 候補がカバーするグリッドセルを計算
    min_grid = (
        int(np.floor(origin[0] / grid_size)),
        int(np.floor(origin[1] / grid_size)),
        int(np.floor(origin[2] / grid_size)),
    )
    max_grid = (
        int(np.floor((origin[0] + size) / grid_size)),
        int(np.floor((origin[1] + size) / grid_size)),
        int(np.floor((origin[2] + size) / grid_size)),
    )

    # 関連するグリッドセル内のセルだけチェック
    checked: set[int] = set()  # セルのid()で重複チェック

    for gx in range(min_grid[0], max_grid[0] + 1):
        for gy in range(min_grid[1], max_grid[1] + 1):
            for gz in range(min_grid[2], max_grid[2] + 1):
                key = (gx, gy, gz)
                if key not in index:
                    continue

                for cell in index[key]:
                    cell_id = id(cell)
                    if cell_id in checked:
                        continue
                    checked.add(cell_id)

                    cell_min = cell.origin
                    cell_max = cell.origin + cell.size

                    # AABBの重なり判定
                    if (
                        candidate_min[0] < cell_max[0] - tolerance
                        and candidate_max[0] > cell_min[0] + tolerance
                        and candidate_min[1] < cell_max[1] - tolerance
                        and candidate_max[1] > cell_min[1] + tolerance
                        and candidate_min[2] < cell_max[2] - tolerance
                        and candidate_max[2] > cell_min[2] + tolerance
                    ):
                        return True

    return False


def _generate_adjacent_external_cells(
    model: VoxelModel,
    occupied_index: dict[tuple[int, int, int], list[Cell]],
    grid_size: float,
    min_cell_size: float,
) -> list[Cell]:
    """占有セルに隣接する外部セル候補を生成する。

    占有セルのサイズが min_cell_size より大きい場合、
    隣接する外部セルを min_cell_size まで分割して生成する。
    """
    candidates: list[Cell] = []
    seen_positions: set[tuple[float, float, float, float]] = set()

    occupied_cells = model.get_occupied_cells()

    for cell in occupied_cells:
        cell_size = cell.size

        # 各方向の隣接位置を計算
        for direction in _NEIGHBOR_DIRECTIONS:
            neighbor_origin = cell.origin + direction * cell_size

            # 隣接セルのサイズはmin_cell_sizeまで分割
            if cell_size > min_cell_size * 1.5:
                # 大きいセルの場合、min_cell_sizeのセルに分割
                sub_cells = _subdivide_to_min_size(neighbor_origin, cell_size, min_cell_size)
                for sub_origin, sub_size in sub_cells:
                    # 元のセルに隣接している部分だけを候補にする
                    if _is_adjacent_to_cell(sub_origin, sub_size, cell):
                        _try_add_candidate(
                            sub_origin,
                            sub_size,
                            occupied_index,
                            grid_size,
                            seen_positions,
                            candidates,
                        )
            else:
                # 同じサイズで隣接セルを生成
                _try_add_candidate(
                    neighbor_origin,
                    cell_size,
                    occupied_index,
                    grid_size,
                    seen_positions,
                    candidates,
                )

    return candidates


def _try_add_candidate(
    origin: np.ndarray,
    size: float,
    occupied_index: dict[tuple[int, int, int], list[Cell]],
    grid_size: float,
    seen_positions: set[tuple[float, float, float, float]],
    candidates: list[Cell],
) -> None:
    """候補セルを追加する（重複・既存占有チェック付き）。"""
    key = (origin[0], origin[1], origin[2], size)

    if key in seen_positions:
        return

    seen_positions.add(key)

    # 既存の占有セルと重なっていないかチェック
    if _is_position_overlapping_any(occupied_index, grid_size, origin, size):
        return

    cell = Cell(
        origin=origin.copy(),
        size=size,
        depth=-1,  # 膨張セルはdepth不定
        state=CellState.EMPTY,  # 後でOUTSIDEに変更
        facet_indices=None,
    )
    candidates.append(cell)


def _subdivide_to_min_size(
    origin: np.ndarray,
    size: float,
    min_size: float,
) -> list[tuple[np.ndarray, float]]:
    """指定した領域をmin_sizeまで8分木で分割する。"""
    result: list[tuple[np.ndarray, float]] = []

    if size <= min_size * 1.5:
        result.append((origin.copy(), size))
        return result

    # 8分割
    half = size / 2.0
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                child_origin = origin + np.array([ix * half, iy * half, iz * half])
                result.extend(_subdivide_to_min_size(child_origin, half, min_size))

    return result


def _is_adjacent_to_cell(
    origin: np.ndarray,
    size: float,
    cell: Cell,
    tolerance: float = 1e-9,
) -> bool:
    """指定した領域がセルに隣接しているかチェックする。

    面で接触している場合にTrueを返す。
    """
    min1 = origin
    max1 = origin + size
    min2 = cell.origin
    max2 = cell.origin + cell.size

    # 各軸方向で隣接判定
    # X方向で隣接
    x_adjacent = abs(max1[0] - min2[0]) < tolerance or abs(min1[0] - max2[0]) < tolerance
    x_overlap = min1[0] < max2[0] - tolerance and max1[0] > min2[0] + tolerance

    # Y方向で隣接
    y_adjacent = abs(max1[1] - min2[1]) < tolerance or abs(min1[1] - max2[1]) < tolerance
    y_overlap = min1[1] < max2[1] - tolerance and max1[1] > min2[1] + tolerance

    # Z方向で隣接
    z_adjacent = abs(max1[2] - min2[2]) < tolerance or abs(min1[2] - max2[2]) < tolerance
    z_overlap = min1[2] < max2[2] - tolerance and max1[2] > min2[2] + tolerance

    # 1軸で隣接し、他の2軸で重なっている場合に隣接
    if x_adjacent and y_overlap and z_overlap:
        return True
    if y_adjacent and x_overlap and z_overlap:
        return True
    if z_adjacent and x_overlap and y_overlap:
        return True

    return False


def _validate_cells_by_distance(
    candidates: list[Cell],
    dist_calc: MeshDistanceCalculator,
    expand_distance: float,
) -> list[Cell]:
    """候補セルの頂点の距離を検証し、条件を満たすものだけ返す。

    セルの全8頂点がexpand_distance以上の距離を持つ場合のみ承認する。
    """
    validated: list[Cell] = []

    for cell in candidates:
        vertices = _get_cell_vertices(cell)
        distances = dist_calc.distances_batch_with_hint(vertices, expand_distance)

        if np.all(distances >= expand_distance):
            validated.append(cell)

    return validated


def _get_cell_vertices(cell: Cell) -> np.ndarray:
    """セルの8頂点を取得する。"""
    origin = cell.origin
    size = cell.size
    vertices = np.array(
        [
            origin + [0, 0, 0],
            origin + [size, 0, 0],
            origin + [0, size, 0],
            origin + [size, size, 0],
            origin + [0, 0, size],
            origin + [size, 0, size],
            origin + [0, size, size],
            origin + [size, size, size],
        ]
    )
    return vertices
