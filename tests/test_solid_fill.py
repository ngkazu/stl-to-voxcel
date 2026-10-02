from __future__ import annotations

import numpy as np

from core.octree import Voxel
from core.solid_fill import fill_solid


def _cube_surface_voxels(lo: int, hi: int, cell_size: float = 1.0) -> list[Voxel]:
    """indices lo..hi(両端含む)の立方体の表面セルをVoxelのリストとして生成する。"""
    voxels = []
    for x in range(lo, hi + 1):
        for y in range(lo, hi + 1):
            for z in range(lo, hi + 1):
                if x in (lo, hi) or y in (lo, hi) or z in (lo, hi):
                    min_corner = np.array([x, y, z], dtype=np.float64) * cell_size
                    voxels.append(Voxel(min_corner, min_corner + cell_size))
    return voxels


def test_fill_solid_simple_cube_fills_interior() -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([6.0, 6.0, 6.0])  # 6x6x6グリッド(index 0..5)、index 0,5は外側padding
    shell = _cube_surface_voxels(1, 4)  # 表面は index 1 と 4

    fill_voxels = fill_solid(shell, bbox_min, bbox_max, cell_size=1.0)

    # 内部の2x2x2(index 2,3)がすべて埋まるはず
    assert len(fill_voxels) == 8
    for voxel in fill_voxels:
        index = np.round(voxel.min_corner).astype(int)
        assert np.all((index >= 2) & (index <= 3))


def test_fill_solid_hollow_shell_leaves_inner_cavity_empty() -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([9.0, 9.0, 9.0])  # 9x9x9グリッド(index 0..8)
    outer_shell = _cube_surface_voxels(1, 7)  # 外殻（index 0,8は空気のpadding）
    inner_shell = _cube_surface_voxels(3, 5)  # 内殻（index 3..5の表面）

    fill_voxels = fill_solid(outer_shell + inner_shell, bbox_min, bbox_max, cell_size=1.0)

    fill_indices = {tuple(np.round(v.min_corner).astype(int)) for v in fill_voxels}

    assert (2, 2, 2) in fill_indices  # 外殻と内殻の間（材質）は埋まる
    assert (4, 4, 4) not in fill_indices  # 内殻の中心（入れ子の空洞）は埋まらない
    assert (0, 0, 0) not in fill_indices  # 外側のpadding領域はそもそも対象外
