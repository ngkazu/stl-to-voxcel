"""Voxel群の膨張処理と距離検証。

STL表面からユーザー指定の距離以上離れていることを保証するための処理を行う。
将来的には収縮（STL表面への密着）機能の追加も想定している。

高速化のポイント:
- Voxelに記録されたfacet_indicesを活用し、距離計算対象を限定
- 膨張後のセルには近傍から面番号を伝播させる
- 距離計算をNumPyでベクトル化
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from core.geometry import point_to_triangle_distance
from core.octree import Voxel

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


def expand_and_validate(
    voxels: list[Voxel],
    occupancy: np.ndarray,
    origin: np.ndarray,
    cell_size: float,
    facets: np.ndarray,
    expand_distance: float,
) -> np.ndarray:
    """Voxel群を膨張させ、STL表面からの距離を検証してフィルタリングする。

    Args:
        voxels: 元のVoxelリスト（facet_indices付き）
        occupancy: 占有グリッド (bool配列)
        origin: グリッドの原点座標
        cell_size: セルサイズ
        facets: 元STLの三角形群 (N, 3, 3)
        expand_distance: STL表面から離す距離

    Returns:
        検証済みの占有グリッド (bool配列)
    """
    # 1. Voxelからセル位置→面番号のマッピングを構築
    cell_to_facets = _build_cell_to_facets_map(voxels, origin, cell_size, occupancy.shape)

    # 2. 膨張処理
    expanded = _dilate_occupancy(occupancy, cell_size, expand_distance)

    # 3. 膨張後の新セルに面番号を伝播
    cell_to_facets = _propagate_facets_to_expanded(
        expanded, occupancy, cell_to_facets, cell_size, expand_distance
    )

    # 4. 外殻頂点を抽出（関連セル情報付き）
    exterior_vertices, vertex_to_cells = _extract_exterior_vertices(expanded, origin, cell_size)

    if len(exterior_vertices) == 0:
        return expanded

    # 5. 各外殻頂点から元STLへの距離を計算（面番号を活用した高速版）
    distances = _compute_distances_fast(exterior_vertices, vertex_to_cells, cell_to_facets, facets)

    # 6. 距離が不足している頂点を特定
    insufficient_mask = distances < expand_distance

    # 7. 距離不足の頂点を持つセルを除外
    validated = _filter_by_distance(expanded, exterior_vertices, vertex_to_cells, insufficient_mask)

    return validated


def _build_cell_to_facets_map(
    voxels: list[Voxel],
    origin: np.ndarray,
    cell_size: float,
    grid_shape: tuple[int, ...],
) -> dict[tuple[int, int, int], set[int]]:
    """Voxelリストからセル位置→面番号のマッピングを構築する。"""
    cell_to_facets: dict[tuple[int, int, int], set[int]] = {}
    grid_shape_arr = np.array(grid_shape)

    for voxel in voxels:
        if not voxel.facet_indices:
            continue

        # Voxelの位置をグリッドインデックスに変換
        lo = np.clip(
            np.floor((voxel.min_corner - origin) / cell_size).astype(int),
            0,
            grid_shape_arr - 1,
        )
        hi = np.clip(
            np.ceil((voxel.max_corner - origin) / cell_size).astype(int),
            1,
            grid_shape_arr,
        )

        # このVoxelがカバーする全セルに面番号を登録
        for ix in range(lo[0], hi[0]):
            for iy in range(lo[1], hi[1]):
                for iz in range(lo[2], hi[2]):
                    cell_key = (ix, iy, iz)
                    if cell_key not in cell_to_facets:
                        cell_to_facets[cell_key] = set()
                    cell_to_facets[cell_key].update(voxel.facet_indices)

    return cell_to_facets


def _propagate_facets_to_expanded(
    expanded: np.ndarray,
    original: np.ndarray,
    cell_to_facets: dict[tuple[int, int, int], set[int]],
    cell_size: float,
    expand_distance: float,
) -> dict[tuple[int, int, int], set[int]]:
    """膨張で追加されたセルに、近傍セルから面番号を伝播させる。

    膨張量に応じた範囲の近傍を探索し、その面番号をマージする。
    """
    result = dict(cell_to_facets)  # 元のマッピングをコピー

    # 膨張で追加されたセル
    new_cells = expanded & ~original
    new_indices = np.argwhere(new_cells)

    if len(new_indices) == 0:
        return result

    # 探索範囲: 膨張量 + 1セル分の余裕
    search_radius = int(np.ceil(expand_distance / cell_size)) + 1
    shape = np.array(expanded.shape)

    # 近傍オフセットを事前計算（球状の近傍）
    offsets = []
    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            for dz in range(-search_radius, search_radius + 1):
                if dx * dx + dy * dy + dz * dz <= search_radius * search_radius:
                    offsets.append((dx, dy, dz))
    offsets = np.array(offsets, dtype=np.int32)

    for idx in new_indices:
        ix, iy, iz = idx
        cell_key = (ix, iy, iz)

        # 近傍セルの面番号を収集
        facets_for_cell: set[int] = set()
        neighbors = idx + offsets
        # 境界内に収める
        valid = np.all((neighbors >= 0) & (neighbors < shape), axis=1)
        neighbors = neighbors[valid]

        for n in neighbors:
            n_key = (int(n[0]), int(n[1]), int(n[2]))
            if n_key in cell_to_facets:
                facets_for_cell.update(cell_to_facets[n_key])

        if facets_for_cell:
            result[cell_key] = facets_for_cell

    return result


def _dilate_occupancy(
    occupancy: np.ndarray,
    cell_size: float,
    expand_distance: float,
) -> np.ndarray:
    """占有グリッドをモルフォロジー膨張させる。"""
    expand_cells = int(np.ceil(expand_distance / cell_size))

    if expand_cells <= 0:
        return occupancy.copy()

    # 6連結の構造要素で膨張
    struct = ndimage.generate_binary_structure(3, 1)
    expanded = ndimage.binary_dilation(
        occupancy,
        structure=struct,
        iterations=expand_cells,
    )

    return expanded


def _extract_exterior_vertices(
    occupancy: np.ndarray,
    origin: np.ndarray,
    cell_size: float,
) -> tuple[np.ndarray, dict[int, list[tuple[int, int, int]]]]:
    """外殻に露出している頂点を抽出する。

    Returns:
        (外殻頂点の座標配列, 頂点インデックス→セルインデックスのマッピング)
    """
    shape = np.array(occupancy.shape)

    # 頂点座標→(頂点インデックス, セルリスト) のマッピング
    vertex_coord_to_info: dict[tuple[float, ...], tuple[int, list[tuple[int, int, int]]]] = {}
    next_vertex_idx = 0

    # 占有セルを走査
    occupied_indices = np.argwhere(occupancy)

    for idx in occupied_indices:
        ix, iy, iz = idx

        # 6方向の隣接セルをチェック
        for offset in _NEIGHBOR_OFFSETS_6:
            neighbor = idx + offset
            direction = tuple(offset)

            # 境界外または空セルに隣接 → この面は外に露出
            is_exterior = (
                np.any(neighbor < 0) or np.any(neighbor >= shape) or not occupancy[tuple(neighbor)]
            )

            if is_exterior:
                # この面の4頂点を追加
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

    # 座標配列とインデックス→セルマッピングを構築
    vertices = np.zeros((len(vertex_coord_to_info), 3), dtype=np.float64)
    vertex_to_cells: dict[int, list[tuple[int, int, int]]] = {}

    for coord, (vidx, cells) in vertex_coord_to_info.items():
        vertices[vidx] = coord
        vertex_to_cells[vidx] = cells

    return vertices, vertex_to_cells


def _compute_distances_fast(
    vertices: np.ndarray,
    vertex_to_cells: dict[int, list[tuple[int, int, int]]],
    cell_to_facets: dict[tuple[int, int, int], set[int]],
    facets: np.ndarray,
) -> np.ndarray:
    """面番号のヒントを活用して距離計算を高速化する。

    各頂点について、関連するセルが持つ面番号だけをチェックする。
    """
    n_vertices = len(vertices)
    distances = np.full(n_vertices, float("inf"), dtype=np.float64)

    for vidx in range(n_vertices):
        vertex = vertices[vidx]

        # この頂点に関連するセルを取得
        related_cells = vertex_to_cells.get(vidx, [])

        # 候補となる面番号を収集（関連セルのすべての面番号をマージ）
        candidate_facets: set[int] = set()
        for cell in related_cells:
            if cell in cell_to_facets:
                candidate_facets.update(cell_to_facets[cell])

        # 候補面がない場合は全面をチェック（フォールバック - 稀なケース）
        if not candidate_facets:
            candidate_facets = set(range(len(facets)))

        # 候補面だけで距離計算（NumPyでベクトル化）
        candidate_list = list(candidate_facets)
        candidate_triangles = facets[candidate_list]  # shape: (M, 3, 3)

        min_dist = _point_to_triangles_min_distance(vertex, candidate_triangles)
        distances[vidx] = min_dist

    return distances


def _point_to_triangles_min_distance(point: np.ndarray, triangles: np.ndarray) -> float:
    """点から複数の三角形への最短距離を計算する（ベクトル化版）。

    Args:
        point: 点の座標 (3,)
        triangles: 三角形群 (M, 3, 3)

    Returns:
        最短距離
    """
    if len(triangles) == 0:
        return float("inf")

    # 小規模なら逐次計算の方が効率的
    if len(triangles) <= 10:
        return min(point_to_triangle_distance(point, tri) for tri in triangles)

    # 大規模ならベクトル化
    point = np.asarray(point, dtype=np.float64)
    v0 = triangles[:, 0]  # (M, 3)
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]

    edge0 = v1 - v0  # (M, 3)
    edge1 = v2 - v0
    v0_to_point = point - v0  # (M, 3)

    # 重心座標を計算
    d00 = np.einsum("ij,ij->i", edge0, edge0)  # (M,)
    d01 = np.einsum("ij,ij->i", edge0, edge1)
    d11 = np.einsum("ij,ij->i", edge1, edge1)
    d20 = np.einsum("ij,ij->i", v0_to_point, edge0)
    d21 = np.einsum("ij,ij->i", v0_to_point, edge1)

    denom = d00 * d11 - d01 * d01
    # 退化三角形は後で個別処理
    valid = np.abs(denom) > 1e-12
    u = np.zeros(len(triangles))
    v = np.zeros(len(triangles))
    u[valid] = (d11[valid] * d20[valid] - d01[valid] * d21[valid]) / denom[valid]
    v[valid] = (d00[valid] * d21[valid] - d01[valid] * d20[valid]) / denom[valid]
    w = 1.0 - u - v

    # 三角形内部に投影されるかどうか
    inside = valid & (u >= 0) & (v >= 0) & (w >= 0)

    distances = np.full(len(triangles), float("inf"))

    # 内部に投影される場合
    if np.any(inside):
        closest_inside = (
            v0[inside] + u[inside, None] * edge0[inside] + v[inside, None] * edge1[inside]
        )
        distances[inside] = np.linalg.norm(point - closest_inside, axis=1)

    # 外部の場合は辺への距離を計算（逐次処理にフォールバック）
    outside_indices = np.where(~inside)[0]
    for idx in outside_indices:
        distances[idx] = point_to_triangle_distance(point, triangles[idx])

    return float(np.min(distances))


def _filter_by_distance(
    occupancy: np.ndarray,
    vertices: np.ndarray,
    vertex_to_cells: dict[int, list[tuple[int, int, int]]],
    insufficient_mask: np.ndarray,
) -> np.ndarray:
    """距離が不足している頂点を持つセルを除外する。"""
    result = occupancy.copy()

    # 距離不足の頂点を持つセルを収集
    cells_to_remove: set[tuple[int, int, int]] = set()

    insufficient_indices = np.where(insufficient_mask)[0]
    for vidx in insufficient_indices:
        if vidx in vertex_to_cells:
            cells_to_remove.update(vertex_to_cells[vidx])

    # セルを除外
    for cell in cells_to_remove:
        result[cell] = False

    return result


def occupancy_to_voxels(
    occupancy: np.ndarray,
    origin: np.ndarray,
    cell_size: float,
) -> list[Voxel]:
    """占有グリッドからVoxelリストを生成する。

    Args:
        occupancy: 占有グリッド
        origin: グリッドの原点座標
        cell_size: セルサイズ

    Returns:
        Voxelのリスト
    """
    voxels: list[Voxel] = []
    for idx in np.argwhere(occupancy):
        min_corner = origin + idx * cell_size
        max_corner = min_corner + cell_size
        voxels.append(Voxel(min_corner, max_corner))
    return voxels
