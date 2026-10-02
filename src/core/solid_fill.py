"""シェルVoxelから内部充填（ソリッド化）を行う。外側からのフラッドフィル+パリティ判定。

入れ子になったシェル（例: 外殻+内殻を持つ中空モデル）がある場合、単純に
「外側から到達できない空セル=ソリッド」とすると、内部の空洞まで誤って埋めてしまう。
そこで、空セルを6連結の連結成分（領域）に分け、領域同士がシェルセルを挟んで隣接する
関係をグラフとして辿り、外側からの「シェル通過回数」の偶奇（パリティ）を求める。
奇数回通過した領域=材質（ソリッド）、偶数回（0を除く）=入れ子の空洞（空気のまま）とする。
"""

from __future__ import annotations

from collections import deque

import numpy as np

from core.octree import Voxel

_UNLABELED = -1
_NEIGHBOR_OFFSETS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def fill_solid(
    voxels: list[Voxel],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    cell_size: float,
) -> list[Voxel]:
    """シェルVoxel群から、奇数パリティ（材質）と判定されたセルのVoxelを新たに生成して返す。

    戻り値には元のシェルVoxelは含まない（内部充填セルのみ）。偶数パリティ（入れ子の空洞）は
    空気として扱われ、結果に含まれない。
    """
    occupancy, origin = _rasterize_shell(voxels, bbox_min, bbox_max, cell_size)
    region_labels, num_regions = _label_empty_regions(occupancy)
    if num_regions == 0:
        return []

    outside_region = _find_outside_region(region_labels)
    parity = _compute_region_parity(occupancy, region_labels, num_regions, outside_region)
    solid_regions = {region for region, p in parity.items() if p % 2 == 1}

    fill_voxels: list[Voxel] = []
    for index in np.argwhere(np.isin(region_labels, list(solid_regions))):
        cell_min = origin + index * cell_size
        fill_voxels.append(Voxel(cell_min, cell_min + cell_size))
    return fill_voxels


def _rasterize_shell(
    voxels: list[Voxel],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """シェルVoxel群を、一様格子のbool占有グリッド（True=シェル）にラスタライズする。

    bboxの外周に1セル分のpaddingを追加する。「外側」判定はグリッド境界面の空セルを
    頼りにしており、bboxがモデルにぴったり密着している（余白がない）とそれが成立しないため。
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64) - cell_size
    bbox_max = np.asarray(bbox_max, dtype=np.float64) + cell_size
    shape = np.maximum(np.ceil((bbox_max - bbox_min) / cell_size).astype(int), 1)
    occupancy = np.zeros(tuple(shape), dtype=bool)

    for voxel in voxels:
        lo = np.clip(np.floor((voxel.min_corner - bbox_min) / cell_size).astype(int), 0, shape - 1)
        hi = np.clip(np.ceil((voxel.max_corner - bbox_min) / cell_size).astype(int), 1, shape)
        occupancy[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True

    return occupancy, bbox_min


def _label_empty_regions(occupancy: np.ndarray) -> tuple[np.ndarray, int]:
    """空セル(occupancy=False)を6連結の連結成分に分け、領域ラベルの配列と領域数を返す。"""
    labels = np.full(occupancy.shape, _UNLABELED, dtype=np.int32)
    nx, ny, nz = occupancy.shape
    next_label = 0

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if occupancy[ix, iy, iz] or labels[ix, iy, iz] != _UNLABELED:
                    continue
                _bfs_fill_region(occupancy, labels, (ix, iy, iz), next_label)
                next_label += 1

    return labels, next_label


def _bfs_fill_region(
    occupancy: np.ndarray,
    labels: np.ndarray,
    start: tuple[int, int, int],
    label: int,
) -> None:
    """startを起点に6連結の空セルをBFSで塗りつぶし、labelsに書き込む。"""
    shape = occupancy.shape
    queue: deque[tuple[int, int, int]] = deque([start])
    labels[start] = label

    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            nx_, ny_, nz_ = x + dx, y + dy, z + dz
            if not (0 <= nx_ < shape[0] and 0 <= ny_ < shape[1] and 0 <= nz_ < shape[2]):
                continue
            if occupancy[nx_, ny_, nz_] or labels[nx_, ny_, nz_] != _UNLABELED:
                continue
            labels[nx_, ny_, nz_] = label
            queue.append((nx_, ny_, nz_))


def _find_outside_region(labels: np.ndarray) -> int:
    """グリッド境界面に露出している空セルの領域ラベルを「外側」とみなして返す。"""
    faces = (
        labels[0, :, :],
        labels[-1, :, :],
        labels[:, 0, :],
        labels[:, -1, :],
        labels[:, :, 0],
        labels[:, :, -1],
    )
    for face in faces:
        valid = face[face != _UNLABELED]
        if valid.size > 0:
            return int(valid[0])

    raise ValueError(
        "外側とみなせる空セルが見つかりませんでした。"
        "bboxがモデルに対して狭すぎる可能性があります（--cubic-root の利用を検討してください）。"
    )


def _compute_region_parity(
    occupancy: np.ndarray,
    labels: np.ndarray,
    num_regions: int,
    outside_region: int,
) -> dict[int, int]:
    """シェルセルを挟んで隣接する領域同士をグラフでつなぎ、外側からのパリティ(偶奇)を求める。"""
    adjacency: dict[int, set[int]] = {region: set() for region in range(num_regions)}
    shape = occupancy.shape

    for x, y, z in np.argwhere(occupancy):
        neighbor_regions: set[int] = set()
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            nx_, ny_, nz_ = x + dx, y + dy, z + dz
            if not (0 <= nx_ < shape[0] and 0 <= ny_ < shape[1] and 0 <= nz_ < shape[2]):
                continue
            label = labels[nx_, ny_, nz_]
            if label != _UNLABELED:
                neighbor_regions.add(int(label))
        for region_a in neighbor_regions:
            for region_b in neighbor_regions:
                if region_a != region_b:
                    adjacency[region_a].add(region_b)

    parity: dict[int, int] = {outside_region: 0}
    queue: deque[int] = deque([outside_region])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in parity:
                parity[neighbor] = 1 - parity[current]
                queue.append(neighbor)

    # シェルで完全に孤立していて到達できなかった領域は、安全側（材質扱い=奇数）としておく。
    for region in range(num_regions):
        parity.setdefault(region, 1)

    return parity
