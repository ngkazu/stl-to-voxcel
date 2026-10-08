"""シェルVoxelから内部充填（ソリッド化）を行う。外側からのフラッドフィル+パリティ判定。

入れ子になったシェル（例: 外殻+内殻を持つ中空モデル）がある場合、単純に
「外側から到達できない空セル=ソリッド」とすると、内部の空洞まで誤って埋めてしまう。
そこで、空セルを6連結の連結成分（領域）に分け、領域同士がシェルセルを挟んで隣接する
関係をグラフとして辿り、外側からの「シェル通過回数」の偶奇（パリティ）を求める。
奇数回通過した領域=材質（ソリッド）、偶数回（0を除く）=入れ子の空洞（空気のまま）とする。

充填セルは locational code で表現する。ラスタライズ格子は 8 分木の最大 depth の
グリッド（モデルの bbox_min を原点、最小セルサイズ単位）に整列させ、グリッド
インデックス (i,j,k) から code を復元する（docs/data_structure.md 参照）。
"""

from __future__ import annotations

from collections import deque

import numpy as np

from core.model import Cell, CellState, VoxelModel, code_from_grid_index

_UNLABELED = -1
_NEIGHBOR_OFFSETS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def fill_solid(model: VoxelModel, cell_size: np.ndarray) -> None:
    """シェルVoxelから内部充填を行い、INSIDEセルをmodelに追加する。

    元のシェルセルはそのまま残り、内部充填セルがINSIDEとして追加される。
    偶数パリティ（入れ子の空洞）は空気として扱われ、追加されない。

    Args:
        model: VoxelModel（SHELLセルが登録済み）
        cell_size: 充填セルのサイズ [sx, sy, sz]（= 最大 depth のセルサイズ）
    """
    cell_size = np.asarray(cell_size, dtype=np.float64)
    shell_cells = model.get_shell_cells()
    if not shell_cells:
        return

    # 充填格子の depth（最小セルサイズ = base_cell_size / 2**depth）
    fill_depth = model.max_depth()

    # ラスタライズ: モデルの bbox_min を基準に、1セル分のパディングを加えた格子を作る。
    # グリッドは 8分木の最大 depth グリッドに整列している（原点は bbox_min - 1セル）。
    occupancy = _rasterize_shell(model, shell_cells, cell_size)

    region_labels, num_regions = _label_empty_regions(occupancy)
    if num_regions == 0:
        return

    outside_region = _find_outside_region(region_labels)
    parity = _compute_region_parity(occupancy, region_labels, num_regions, outside_region)
    solid_regions = {region for region, p in parity.items() if p % 2 == 1}

    # INSIDEセルを追加
    # occupancy のインデックスは [パディング1セル] オフセットされているため、
    # 8分木グリッドインデックスに戻してから code を計算する。
    grid_dim = 2**fill_depth
    for index in np.argwhere(np.isin(region_labels, list(solid_regions))):
        # パディング分(-1)を引いて 8分木グリッド座標に変換
        gi = int(index[0]) - 1
        gj = int(index[1]) - 1
        gk = int(index[2]) - 1
        # 8分木グリッドの範囲外（パディング領域）はスキップ
        if not (0 <= gi < grid_dim and 0 <= gj < grid_dim and 0 <= gk < grid_dim):
            continue
        code = code_from_grid_index((gi, gj, gk), fill_depth)
        cell = Cell(
            code=code,
            state=CellState.INSIDE,
            facet_indices=None,
        )
        model.add_cell(cell)


def _rasterize_shell(
    model: VoxelModel,
    cells: list[Cell],
    cell_size: np.ndarray,
) -> np.ndarray:
    """シェルセル群を、8分木グリッドに整列した占有グリッド（True=シェル）にする。

    グリッドは 8分木の最大 depth グリッド（2**depth セル/軸）に、外周1セルの
    パディングを加えたもの。原点は bbox_min - cell_size（パディング分）。
    「外側」判定はグリッド境界面の空セルを頼りにしているため、パディングが必要。

    occupancy のインデックス (i,j,k) と 8分木グリッド (gi,gj,gk) の関係:
        gi = i - 1  （パディング1セル分のオフセット）
    """
    cell_size = np.asarray(cell_size, dtype=np.float64)
    fill_depth = model.max_depth()
    grid_dim = 2**fill_depth

    # パディング1セルを含む格子形状（全軸 grid_dim + 2）
    shape = np.array([grid_dim + 2, grid_dim + 2, grid_dim + 2], dtype=int)
    occupancy = np.zeros(tuple(shape), dtype=bool)

    # 各 SHELL セルを、その depth に応じて占有グリッドに塗る。
    # セルは最大 depth とは限らない（大きいセルは複数グリッドセルを覆う）。
    bbox_min = np.asarray(model.bbox_min, dtype=np.float64)
    for cell in cells:
        origin, size = model.cell_origin_size(cell)
        # 8分木グリッド座標（最大 depth 基準）での範囲
        lo = np.floor((origin - bbox_min) / cell_size + 1e-9).astype(int)
        hi = np.ceil((origin + size - bbox_min) / cell_size - 1e-9).astype(int)
        lo = np.clip(lo, 0, grid_dim)
        hi = np.clip(hi, 0, grid_dim)
        # パディング分(+1)オフセットして占有を立てる
        occupancy[
            lo[0] + 1 : hi[0] + 1,
            lo[1] + 1 : hi[1] + 1,
            lo[2] + 1 : hi[2] + 1,
        ] = True

    return occupancy


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
