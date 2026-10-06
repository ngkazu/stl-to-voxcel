"""Voxel群の膨張処理と距離検証。

STL表面からユーザー指定の距離以上離れていることを保証するための処理を行う。
将来的には収縮（STL表面への密着）機能の追加も想定している。

高速化のポイント:
- Cellに記録されたfacet_indicesを活用し、距離計算対象を限定
- 膨張後のセルには近傍から面番号を伝播させる
- 距離計算をNumPyでベクトル化
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from core.geometry import point_to_triangle_distance
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
) -> None:
    """VoxelModelを膨張させ、OUTSIDEセルを追加する。

    STL表面からの距離を検証し、距離が不足するセルは追加しない。

    Args:
        model: VoxelModel（SHELL/INSIDEセルが登録済み）
        facets: 元STLの三角形群 (N, 3, 3)
        expand_distance: STL表面から離す距離
        cell_size: 膨張セルのサイズ
    """
    occupied_cells = model.get_occupied_cells()
    if not occupied_cells:
        return

    # bboxを計算
    all_min = np.min([c.min_corner for c in occupied_cells], axis=0)
    all_max = np.max([c.max_corner for c in occupied_cells], axis=0)

    # 占有グリッドを作成
    occupancy, origin = _rasterize_cells(occupied_cells, all_min, all_max, cell_size)

    # セル位置→面番号のマッピングを構築
    cell_to_facets = _build_cell_to_facets_map(occupied_cells, origin, cell_size, occupancy.shape)

    # 膨張処理
    expanded = _dilate_occupancy(occupancy, cell_size, expand_distance)

    # 膨張後の新セルに面番号を伝播
    cell_to_facets = _propagate_facets_to_expanded(
        expanded, occupancy, cell_to_facets, cell_size, expand_distance
    )

    # 外殻頂点を抽出
    exterior_vertices, vertex_to_cells = _extract_exterior_vertices(expanded, origin, cell_size)

    if len(exterior_vertices) == 0:
        return

    # 距離計算
    distances = _compute_distances_fast(exterior_vertices, vertex_to_cells, cell_to_facets, facets)

    # 距離が不足している頂点を特定
    insufficient_mask = distances < expand_distance

    # 距離不足の頂点を持つセルを除外
    validated = _filter_by_distance(expanded, exterior_vertices, vertex_to_cells, insufficient_mask)

    # 新規追加されたセル（元のoccupancyにはなかったセル）をOUTSIDEとして追加
    new_cells_mask = validated & ~occupancy
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


def _rasterize_cells(
    cells: list[Cell],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """セル群を一様格子のbool占有グリッドにラスタライズする。"""
    bbox_min = np.asarray(bbox_min, dtype=np.float64) - cell_size
    bbox_max = np.asarray(bbox_max, dtype=np.float64) + cell_size
    shape = np.maximum(np.ceil((bbox_max - bbox_min) / cell_size).astype(int), 1)
    occupancy = np.zeros(tuple(shape), dtype=bool)

    for cell in cells:
        lo = np.clip(np.floor((cell.min_corner - bbox_min) / cell_size).astype(int), 0, shape - 1)
        hi = np.clip(np.ceil((cell.max_corner - bbox_min) / cell_size).astype(int), 1, shape)
        occupancy[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True

    return occupancy, bbox_min


def _build_cell_to_facets_map(
    cells: list[Cell],
    origin: np.ndarray,
    cell_size: float,
    grid_shape: tuple[int, ...],
) -> dict[tuple[int, int, int], set[int]]:
    """セルリストからグリッド位置→面番号のマッピングを構築する。"""
    cell_to_facets: dict[tuple[int, int, int], set[int]] = {}
    grid_shape_arr = np.array(grid_shape)

    for cell in cells:
        if not cell.facet_indices:
            continue

        lo = np.clip(
            np.floor((cell.min_corner - origin) / cell_size).astype(int),
            0,
            grid_shape_arr - 1,
        )
        hi = np.clip(
            np.ceil((cell.max_corner - origin) / cell_size).astype(int),
            1,
            grid_shape_arr,
        )

        for ix in range(lo[0], hi[0]):
            for iy in range(lo[1], hi[1]):
                for iz in range(lo[2], hi[2]):
                    cell_key = (ix, iy, iz)
                    if cell_key not in cell_to_facets:
                        cell_to_facets[cell_key] = set()
                    cell_to_facets[cell_key].update(cell.facet_indices)

    return cell_to_facets


def _propagate_facets_to_expanded(
    expanded: np.ndarray,
    original: np.ndarray,
    cell_to_facets: dict[tuple[int, int, int], set[int]],
    cell_size: float,
    expand_distance: float,
) -> dict[tuple[int, int, int], set[int]]:
    """膨張で追加されたセルに、近傍セルから面番号を伝播させる。"""
    result = dict(cell_to_facets)

    new_cells = expanded & ~original
    new_indices = np.argwhere(new_cells)

    if len(new_indices) == 0:
        return result

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

        facets_for_cell: set[int] = set()
        neighbors = idx + offsets
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


def _compute_distances_fast(
    vertices: np.ndarray,
    vertex_to_cells: dict[int, list[tuple[int, int, int]]],
    cell_to_facets: dict[tuple[int, int, int], set[int]],
    facets: np.ndarray,
) -> np.ndarray:
    """面番号のヒントを活用して距離計算を高速化する。"""
    n_vertices = len(vertices)
    distances = np.full(n_vertices, float("inf"), dtype=np.float64)

    for vidx in range(n_vertices):
        vertex = vertices[vidx]

        related_cells = vertex_to_cells.get(vidx, [])

        candidate_facets: set[int] = set()
        for cell in related_cells:
            if cell in cell_to_facets:
                candidate_facets.update(cell_to_facets[cell])

        if not candidate_facets:
            candidate_facets = set(range(len(facets)))

        candidate_list = list(candidate_facets)
        candidate_triangles = facets[candidate_list]

        min_dist = _point_to_triangles_min_distance(vertex, candidate_triangles)
        distances[vidx] = min_dist

    return distances


def _point_to_triangles_min_distance(point: np.ndarray, triangles: np.ndarray) -> float:
    """点から複数の三角形への最短距離を計算する（ベクトル化版）。"""
    if len(triangles) == 0:
        return float("inf")

    if len(triangles) <= 10:
        return min(point_to_triangle_distance(point, tri) for tri in triangles)

    point = np.asarray(point, dtype=np.float64)
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]

    edge0 = v1 - v0
    edge1 = v2 - v0
    v0_to_point = point - v0

    d00 = np.einsum("ij,ij->i", edge0, edge0)
    d01 = np.einsum("ij,ij->i", edge0, edge1)
    d11 = np.einsum("ij,ij->i", edge1, edge1)
    d20 = np.einsum("ij,ij->i", v0_to_point, edge0)
    d21 = np.einsum("ij,ij->i", v0_to_point, edge1)

    denom = d00 * d11 - d01 * d01
    valid = np.abs(denom) > 1e-12
    u = np.zeros(len(triangles))
    v = np.zeros(len(triangles))
    u[valid] = (d11[valid] * d20[valid] - d01[valid] * d21[valid]) / denom[valid]
    v[valid] = (d00[valid] * d21[valid] - d01[valid] * d20[valid]) / denom[valid]
    w = 1.0 - u - v

    inside = valid & (u >= 0) & (v >= 0) & (w >= 0)

    distances = np.full(len(triangles), float("inf"))

    if np.any(inside):
        closest_inside = (
            v0[inside] + u[inside, None] * edge0[inside] + v[inside, None] * edge1[inside]
        )
        distances[inside] = np.linalg.norm(point - closest_inside, axis=1)

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

    cells_to_remove: set[tuple[int, int, int]] = set()

    insufficient_indices = np.where(insufficient_mask)[0]
    for vidx in insufficient_indices:
        if vidx in vertex_to_cells:
            cells_to_remove.update(vertex_to_cells[vidx])

    for cell in cells_to_remove:
        result[cell] = False

    return result
