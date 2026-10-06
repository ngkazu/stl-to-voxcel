"""Voxel群の膨張処理と距離検証。

STL表面からユーザー指定の距離以上離れていることを保証するための処理を行う。
将来的には収縮（STL表面への密着）機能の追加も想定している。

高速化のポイント:
- KD-treeを使った近傍三角形検索で距離計算を高速化
- ヒント半径を使った早期終了で不要な計算を削減
- 距離不足の箇所だけを選択的に膨張（不要な膨張を回避）
"""

from __future__ import annotations

import numpy as np

from core.geometry import MeshDistanceCalculator
from core.model import Cell, CellState, VoxelModel

# 6方向の隣接オフセット
_NEIGHBOR_OFFSETS_6 = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=np.int32,
)

# 各方向に対応する面の4頂点オフセット（セル座標からの相対位置）
# セルの min_corner を基準に、各面の4頂点を定義
_FACE_VERTEX_OFFSETS = {
    (1, 0, 0): np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]),  # +X面
    (-1, 0, 0): np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]),  # -X面
    (0, 1, 0): np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]]),  # +Y面
    (0, -1, 0): np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]]),  # -Y面
    (0, 0, 1): np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]),  # +Z面
    (0, 0, -1): np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]),  # -Z面
}


def expand_model(
    model: VoxelModel,
    facets: np.ndarray,
    expand_distance: float,
    cell_size: float,
    max_iterations: int = 100,
) -> None:
    """VoxelModelを膨張させ、OUTSIDEセルを追加する。

    外殻頂点がSTL表面から expand_distance 以上離れることを保証する。
    既に条件を満たしている場合は膨張しない。

    **選択的膨張アルゴリズム:**
    全体をモルフォロジー膨張するのではなく、距離不足の箇所だけを選択的に膨張する。
    1. 外殻頂点の距離を計算
    2. expand_distance 未満の頂点を持つセルを特定
    3. それらのセルの隣接セル（未占有）だけを膨張候補にする
    4. 候補セルの距離を検証し、条件を満たすものだけ追加
    5. 条件を満たすまで繰り返す

    Args:
        model: VoxelModel（SHELL/INSIDEセルが登録済み）
        facets: 元STLの三角形群 (N, 3, 3)
        expand_distance: STL表面から離す距離
        cell_size: 膨張セルのサイズ
        max_iterations: 最大反復回数（無限ループ防止）
    """
    if expand_distance <= 0:
        return

    occupied_cells = model.get_occupied_cells()
    if not occupied_cells:
        return

    # KD-treeを使った距離計算器を準備
    dist_calc = MeshDistanceCalculator(facets)

    # bboxを計算
    all_min = np.min([c.min_corner for c in occupied_cells], axis=0)
    all_max = np.max([c.max_corner for c in occupied_cells], axis=0)

    # 占有グリッドを作成（膨張分のパディングを確保）
    occupancy, origin = _rasterize_cells(
        occupied_cells, all_min, all_max, cell_size, expand_distance
    )
    original_occupancy = occupancy.copy()

    # 反復的に選択的膨張を行う
    for iteration in range(max_iterations):
        # 現在の外殻頂点を抽出
        exterior_vertices, vertex_to_cells = _extract_exterior_vertices(
            occupancy, origin, cell_size
        )

        if len(exterior_vertices) == 0:
            break

        # 距離を計算
        distances = dist_calc.distances_batch_with_hint(exterior_vertices, expand_distance)

        # 距離不足の頂点を特定
        insufficient_mask = distances < expand_distance

        if not np.any(insufficient_mask):
            # 全ての頂点が条件を満たしている → 完了
            break

        # 距離不足の頂点を持つセルを特定
        insufficient_cells = _get_cells_with_insufficient_vertices(
            vertex_to_cells, insufficient_mask
        )

        if not insufficient_cells:
            break

        # それらのセルの隣接セル（未占有）を膨張候補として取得
        candidates = _get_expansion_candidates(occupancy, insufficient_cells)

        if not candidates:
            # 候補がない → これ以上膨張できない
            break

        # 候補セルの全頂点の距離を検証
        validated_candidates = _validate_candidates(
            candidates, occupancy, origin, cell_size, dist_calc, expand_distance
        )

        if not validated_candidates:
            # 条件を満たす候補がない → これ以上膨張できない
            break

        # 検証済み候補を占有グリッドに追加
        for idx in validated_candidates:
            occupancy[idx] = True

    # 新規追加されたセル（元のoccupancyにはなかったセル）をOUTSIDEとして追加
    new_cells_mask = occupancy & ~original_occupancy
    for idx in np.argwhere(new_cells_mask):
        cell_origin = origin + idx * cell_size
        cell = Cell(
            origin=cell_origin,
            size=cell_size,
            depth=-1,  # 膨張セルはdepth不定
            state=CellState.OUTSIDE,
            facet_indices=None,
        )
        model.add_cell(cell)


def _get_cells_with_insufficient_vertices(
    vertex_to_cells: dict[int, list[tuple[int, int, int]]],
    insufficient_mask: np.ndarray,
) -> set[tuple[int, int, int]]:
    """距離不足の頂点を持つセルのインデックスを取得する。"""
    insufficient_cells: set[tuple[int, int, int]] = set()

    insufficient_indices = np.where(insufficient_mask)[0]
    for vidx in insufficient_indices:
        if vidx in vertex_to_cells:
            insufficient_cells.update(vertex_to_cells[vidx])

    return insufficient_cells


def _get_expansion_candidates(
    occupancy: np.ndarray,
    source_cells: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    """距離不足セルの隣接セル（未占有）を膨張候補として取得する。"""
    shape = np.array(occupancy.shape)
    candidates: set[tuple[int, int, int]] = set()

    for cell_idx in source_cells:
        cell_arr = np.array(cell_idx)
        for offset in _NEIGHBOR_OFFSETS_6:
            neighbor = cell_arr + offset
            # 範囲外チェック
            if np.any(neighbor < 0) or np.any(neighbor >= shape):
                continue
            neighbor_tuple = tuple(neighbor)
            # 未占有セルだけが候補
            if not occupancy[neighbor_tuple]:
                candidates.add(neighbor_tuple)

    return candidates


def _validate_candidates(
    candidates: set[tuple[int, int, int]],
    occupancy: np.ndarray,
    origin: np.ndarray,
    cell_size: float,
    dist_calc: MeshDistanceCalculator,
    expand_distance: float,
) -> list[tuple[int, int, int]]:
    """候補セルを追加した場合に、新たに露出する頂点の距離を検証する。

    候補セルを追加すると、そのセルの外殻面の頂点が新たに露出する。
    それらの頂点がexpand_distance以上の距離を持つ場合のみ、候補を承認する。
    """
    validated: list[tuple[int, int, int]] = []
    shape = np.array(occupancy.shape)

    for candidate in candidates:
        candidate_arr = np.array(candidate)

        # 候補セルを追加した場合に新たに露出する頂点を計算
        new_exterior_vertices = []

        for offset in _NEIGHBOR_OFFSETS_6:
            neighbor = candidate_arr + offset
            direction = tuple(offset)

            # 隣接セルが範囲外または未占有の場合、その面が外殻になる
            is_exterior = (
                np.any(neighbor < 0) or np.any(neighbor >= shape) or not occupancy[tuple(neighbor)]
            )

            if is_exterior:
                face_offsets = _FACE_VERTEX_OFFSETS[direction]
                for vertex_offset in face_offsets:
                    vertex_pos = origin + (candidate_arr + vertex_offset) * cell_size
                    new_exterior_vertices.append(vertex_pos)

        if not new_exterior_vertices:
            # 外殻面がない（完全に囲まれている）場合は追加OK
            validated.append(candidate)
            continue

        # 重複を除去
        vertices_array = np.array(new_exterior_vertices)
        unique_vertices = np.unique(vertices_array, axis=0)

        # 距離を計算
        distances = dist_calc.distances_batch_with_hint(unique_vertices, expand_distance)

        # 全ての頂点がexpand_distance以上なら承認
        if np.all(distances >= expand_distance):
            validated.append(candidate)

    return validated


def _rasterize_cells(
    cells: list[Cell],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    cell_size: float,
    expand_distance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """セル群を一様格子のbool占有グリッドにラスタライズする。

    expand_distance が指定された場合、膨張処理で必要な領域を確保するため
    その分のパディングを追加する。
    """
    # パディング: 最低1セル + 膨張距離分
    padding = cell_size + expand_distance
    bbox_min = np.asarray(bbox_min, dtype=np.float64) - padding
    bbox_max = np.asarray(bbox_max, dtype=np.float64) + padding
    shape = np.maximum(np.ceil((bbox_max - bbox_min) / cell_size).astype(int), 1)
    occupancy = np.zeros(tuple(shape), dtype=bool)

    for cell in cells:
        lo = np.clip(np.floor((cell.min_corner - bbox_min) / cell_size).astype(int), 0, shape - 1)
        hi = np.clip(np.ceil((cell.max_corner - bbox_min) / cell_size).astype(int), 1, shape)
        occupancy[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True

    return occupancy, bbox_min


def _extract_exterior_vertices(
    occupancy: np.ndarray,
    origin: np.ndarray,
    cell_size: float,
) -> tuple[np.ndarray, dict[int, list[tuple[int, int, int]]]]:
    """外殻に露出している頂点を抽出する。"""
    shape = np.array(occupancy.shape)

    vertex_coord_to_info: dict[tuple[float, ...], tuple[int, list[tuple[int, int, int]]]] = {}
    next_vertex_idx = 0

    occupied_indices = np.argwhere(occupancy)

    for idx in occupied_indices:
        ix, iy, iz = idx

        for offset in _NEIGHBOR_OFFSETS_6:
            neighbor = idx + offset
            direction = tuple(offset)

            is_exterior = (
                np.any(neighbor < 0) or np.any(neighbor >= shape) or not occupancy[tuple(neighbor)]
            )

            if is_exterior:
                face_offsets = _FACE_VERTEX_OFFSETS[direction]
                for vertex_offset in face_offsets:
                    vertex_pos = origin + (idx + vertex_offset) * cell_size
                    vertex_key = tuple(vertex_pos)

                    if vertex_key not in vertex_coord_to_info:
                        vertex_coord_to_info[vertex_key] = (next_vertex_idx, [])
                        next_vertex_idx += 1

                    vertex_coord_to_info[vertex_key][1].append((ix, iy, iz))

    if not vertex_coord_to_info:
        return np.array([]).reshape(0, 3), {}

    vertices = np.zeros((len(vertex_coord_to_info), 3), dtype=np.float64)
    vertex_to_cells: dict[int, list[tuple[int, int, int]]] = {}

    for coord, (vidx, cells) in vertex_coord_to_info.items():
        vertices[vidx] = coord
        vertex_to_cells[vidx] = cells

    return vertices, vertex_to_cells
