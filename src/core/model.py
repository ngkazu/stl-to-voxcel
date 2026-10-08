"""Voxelデータ構造の定義（locational code ベース）。

Cell: 1つのセル（確定したVoxel）。locational code で位置を表現する。
CellState: セルの状態（EMPTY/SHELL/INSIDE/OUTSIDE）
VoxelModel: Voxelモデル全体を管理するクラス

locational code の詳細は docs/data_structure.md を参照。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class CellState(IntEnum):
    """セルの状態"""

    EMPTY = 0  # 空
    SHELL = 1  # 表面（STLと交差）
    INSIDE = 2  # 内部
    OUTSIDE = 3  # 外部（expand追加分）


# ---------------------------------------------------------------------------
# locational code ヘルパー（方式2: 番兵ビット付き）
#
# code = 1 <c1> <c2> ... <cD>   （各 ci は 3bit の子番号 0..7）
# 先頭の 1 は番兵ビット。ルートは code=1。
# 子番号 octant = (ix << 2) | (iy << 1) | iz
# 詳細は docs/data_structure.md を参照。
# ---------------------------------------------------------------------------

ROOT_CODE = 1


def code_depth(code: int) -> int:
    """locational code から depth を求める（ルート=0）。"""
    return (code.bit_length() - 1) // 3


def code_parent(code: int) -> int:
    """親の code を返す（ルート code=1 のときは 1 を返す）。"""
    if code <= ROOT_CODE:
        return ROOT_CODE
    return code >> 3


def code_child(code: int, octant: int) -> int:
    """子の code を返す（octant: 0..7）。"""
    return (code << 3) | (octant & 0b111)


def octant_from_bits(ix: int, iy: int, iz: int) -> int:
    """各軸の上下(0/1)から子番号 octant を求める。"""
    return ((ix & 1) << 2) | ((iy & 1) << 1) | (iz & 1)


def code_to_origin_size(
    code: int,
    bbox_min: np.ndarray,
    base_cell_size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """locational code から (origin, size) を復元する。

    Args:
        code: locational code
        bbox_min: ルートの origin（STL bbox 最小）
        base_cell_size: ルート1辺 [sx, sy, sz]

    Returns:
        (origin, size): このセルの min_corner とサイズ [sx, sy, sz]
    """
    depth = code_depth(code)
    origin = np.asarray(bbox_min, dtype=np.float64).copy()
    size = np.asarray(base_cell_size, dtype=np.float64).copy()

    # 番兵ビットを除いた経路部分
    path = code & ((1 << (3 * depth)) - 1) if depth > 0 else 0

    for level in range(depth):
        size = size / 2.0
        # 上位階層から順に 3bit ずつ読む
        shift = 3 * (depth - 1 - level)
        octant = (path >> shift) & 0b111
        ix = (octant >> 2) & 1
        iy = (octant >> 1) & 1
        iz = octant & 1
        origin = origin + np.array([ix, iy, iz], dtype=np.float64) * size

    return origin, size


def code_from_grid_index(
    index: tuple[int, int, int] | np.ndarray,
    depth: int,
) -> int:
    """指定 depth の一様グリッドインデックス (i,j,k) から locational code を求める。

    depth での各軸セル数は 2**depth。インデックス i のビット列を上位から
    順に octant に組み立てる（8分木の分割順と整合）。

    Args:
        index: グリッドインデックス (i, j, k)（0 <= i,j,k < 2**depth）
        depth: 対象の depth

    Returns:
        locational code
    """
    i, j, k = int(index[0]), int(index[1]), int(index[2])
    code = ROOT_CODE
    for level in range(depth):
        # 上位ビットから順に取り出す
        shift = depth - 1 - level
        ix = (i >> shift) & 1
        iy = (j >> shift) & 1
        iz = (k >> shift) & 1
        code = code_child(code, octant_from_bits(ix, iy, iz))
    return code


def code_to_grid_index(code: int) -> tuple[int, int, int]:
    """locational code を、その depth での一様グリッドインデックス (i,j,k) に変換する。

    code_from_grid_index の逆変換。
    """
    depth = code_depth(code)
    path = code & ((1 << (3 * depth)) - 1) if depth > 0 else 0
    i = j = k = 0
    for level in range(depth):
        shift = 3 * (depth - 1 - level)
        octant = (path >> shift) & 0b111
        ix = (octant >> 2) & 1
        iy = (octant >> 1) & 1
        iz = octant & 1
        # 上位階層ほど上位ビット
        bit = depth - 1 - level
        i |= ix << bit
        j |= iy << bit
        k |= iz << bit
    return i, j, k


# ---------------------------------------------------------------------------
# Cell / VoxelModel
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """1つのセル（確定したVoxel）。

    位置は locational code で表現する。origin/size/depth は code から
    計算で復元できるため、フィールドとしては持たない。
    """

    code: int  # locational code（方式2）
    state: CellState  # 状態
    facet_indices: tuple[int, ...] | None = None  # 交差するSTL面番号（SHELLのみ）

    @property
    def depth(self) -> int:
        """8分木の分割回数（ルート=0）。code から計算。"""
        return code_depth(self.code)


@dataclass
class VoxelModel:
    """Voxelモデル全体を管理するクラス。

    セルは locational code をキーとする dict で保持する。
    座標復元に必要な bbox_min / base_cell_size はモデルが1つだけ持つ。
    """

    bbox_min: np.ndarray  # ルートの origin（STL bbox 最小）
    base_cell_size: np.ndarray  # ルート1辺 [sx, sy, sz]
    cells: dict[int, Cell] = field(default_factory=dict)  # code -> Cell

    # ---- 座標復元 ----

    def cell_origin_size(self, cell: Cell) -> tuple[np.ndarray, np.ndarray]:
        """セルの (origin, size) を返す。"""
        return code_to_origin_size(cell.code, self.bbox_min, self.base_cell_size)

    def cell_min_corner(self, cell: Cell) -> np.ndarray:
        origin, _ = self.cell_origin_size(cell)
        return origin

    def cell_max_corner(self, cell: Cell) -> np.ndarray:
        origin, size = self.cell_origin_size(cell)
        return origin + size

    # ---- セル取得 ----

    def get_cells_by_state(self, state: CellState) -> list[Cell]:
        """指定した状態のセルを取得"""
        return [cell for cell in self.cells.values() if cell.state == state]

    def get_shell_cells(self) -> list[Cell]:
        """SHELLセルを取得"""
        return self.get_cells_by_state(CellState.SHELL)

    def get_inside_cells(self) -> list[Cell]:
        """INSIDEセルを取得"""
        return self.get_cells_by_state(CellState.INSIDE)

    def get_outside_cells(self) -> list[Cell]:
        """OUTSIDEセルを取得"""
        return self.get_cells_by_state(CellState.OUTSIDE)

    def get_occupied_cells(self) -> list[Cell]:
        """占有セル（EMPTY以外）を取得"""
        return [cell for cell in self.cells.values() if cell.state != CellState.EMPTY]

    # ---- セル追加 ----

    def add_cell(self, cell: Cell) -> None:
        """セルを追加（同じ code は上書き）"""
        self.cells[cell.code] = cell

    def add_cells(self, cells: list[Cell]) -> None:
        """複数のセルを追加"""
        for cell in cells:
            self.cells[cell.code] = cell

    # ---- サイズ情報 ----

    def max_depth(self) -> int:
        """登録セルの最大 depth を返す（セルが無ければ 0）。"""
        if not self.cells:
            return 0
        return max(code_depth(code) for code in self.cells)

    def get_min_cell_size(self) -> np.ndarray:
        """SHELLセルの最小セルサイズ [sx, sy, sz] を返す。

        8分木で分割された最小セル（= 最大 depth）のサイズ（各軸）を返す。
        充填・膨張時の格子セルサイズとして使用する。
        cubic_root無しの場合は非立方体（軸ごとに異なる）になりうる。
        """
        shell_cells = self.get_shell_cells()
        target = shell_cells if shell_cells else self.get_occupied_cells()
        if not target:
            return np.asarray(self.base_cell_size, dtype=np.float64).copy()
        # 最大 depth のセルサイズ = base_cell_size / 2**max_depth
        max_d = max(cell.depth for cell in target)
        return np.asarray(self.base_cell_size, dtype=np.float64) / (2.0**max_d)

    # ---- カウント ----

    @property
    def cell_count(self) -> int:
        """総セル数"""
        return len(self.cells)

    @property
    def shell_count(self) -> int:
        """SHELLセル数"""
        return sum(1 for c in self.cells.values() if c.state == CellState.SHELL)

    @property
    def inside_count(self) -> int:
        """INSIDEセル数"""
        return sum(1 for c in self.cells.values() if c.state == CellState.INSIDE)

    @property
    def outside_count(self) -> int:
        """OUTSIDEセル数"""
        return sum(1 for c in self.cells.values() if c.state == CellState.OUTSIDE)
