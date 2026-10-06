"""Voxelデータ構造の定義。

Cell: 1つのセル（確定したVoxel）
CellState: セルの状態（SHELL/INSIDE/OUTSIDE）
VoxelModel: Voxelモデル全体を管理するクラス
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


@dataclass
class Cell:
    """1つのセル（確定したVoxel）"""

    origin: np.ndarray  # このセルの原点（min_corner）
    size: float  # このセルのサイズ
    depth: int  # 8分木の分割回数
    state: CellState  # 状態
    facet_indices: tuple[int, ...] | None = None  # 交差するSTL面番号（SHELLのみ）

    @property
    def min_corner(self) -> np.ndarray:
        """セルの最小座標（originのエイリアス）"""
        return self.origin

    @property
    def max_corner(self) -> np.ndarray:
        """セルの最大座標"""
        return self.origin + self.size

    @property
    def center(self) -> np.ndarray:
        """セルの中心座標"""
        return self.origin + self.size / 2


@dataclass
class VoxelModel:
    """Voxelモデル全体を管理するクラス"""

    origin: np.ndarray  # グリッド全体の原点
    base_cell_size: float  # depth=0 のセルサイズ
    cells: list[Cell] = field(default_factory=list)  # 確定した全セル

    def get_cells_by_state(self, state: CellState) -> list[Cell]:
        """指定した状態のセルを取得"""
        return [cell for cell in self.cells if cell.state == state]

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
        return [cell for cell in self.cells if cell.state != CellState.EMPTY]

    def add_cell(self, cell: Cell) -> None:
        """セルを追加"""
        self.cells.append(cell)

    def add_cells(self, cells: list[Cell]) -> None:
        """複数のセルを追加"""
        self.cells.extend(cells)

    @property
    def cell_count(self) -> int:
        """総セル数"""
        return len(self.cells)

    @property
    def shell_count(self) -> int:
        """SHELLセル数"""
        return sum(1 for c in self.cells if c.state == CellState.SHELL)

    @property
    def inside_count(self) -> int:
        """INSIDEセル数"""
        return sum(1 for c in self.cells if c.state == CellState.INSIDE)

    @property
    def outside_count(self) -> int:
        """OUTSIDEセル数"""
        return sum(1 for c in self.cells if c.state == CellState.OUTSIDE)
