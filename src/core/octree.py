"""STLファセット群から8分木でVoxelを構築するロジック。"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from core.geometry import triangle_box_overlap_batch
from core.model import Cell, CellState, VoxelModel

logger = logging.getLogger(__name__)

# voxel_size の指定ミスによる無限/過大分割を防ぐための安全上限（再分割の深さ）。
MAX_OCTREE_DEPTH = 20


def make_cubic_bbox(bbox_min: np.ndarray, bbox_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """bboxの中心を保ったまま、最大辺の長さに合わせた立方体のbboxへ拡張する。

    非立方体のbboxをそのままルートにすると、子孫Voxelも同じ縦横比の直方体になり
    軸ごとの分解能が不均一になる（Known Limitations参照）。これを避けたい場合に使う。
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    center = (bbox_min + bbox_max) / 2.0
    half_size = float((bbox_max - bbox_min).max()) / 2.0
    return center - half_size, center + half_size


def build_voxel_model(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    facets: np.ndarray,
    voxel_size: float,
    max_depth: int = MAX_OCTREE_DEPTH,
    cubic_root: bool = False,
) -> VoxelModel:
    """STL全体のbboxを8分木で再帰分割し、VoxelModelを返す。

    cubic_root=True の場合、ルートのbboxを最大辺に合わせた立方体に拡張してから分割する
    （はみ出した領域はファセットと交差しないため、最初の判定で破棄されるだけで済む）。
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)

    if cubic_root:
        bbox_min, bbox_max = make_cubic_bbox(bbox_min, bbox_max)

    # base_cell_size: ルート（depth=0）のセルサイズ
    base_cell_size = float((bbox_max - bbox_min).max())

    model = VoxelModel(
        origin=bbox_min.copy(),
        base_cell_size=base_cell_size,
        cells=[],
    )

    all_facet_indices = np.arange(len(facets))
    queue: deque[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = deque()
    queue.append((bbox_min, bbox_max, all_facet_indices, 0))

    while queue:
        min_corner, max_corner, candidate_indices, depth = queue.popleft()
        for child_min, child_max in _split_octants(min_corner, max_corner):
            hit_indices = _intersecting_facets(child_min, child_max, facets, candidate_indices)
            if len(hit_indices) == 0:
                continue  # flag0: 交差なし -> 破棄

            child_depth = depth + 1
            # 各軸のサイズ [sx, sy, sz]（cubic_root無しだと非立方体になりうる）
            child_size = child_max - child_min
            # 分割判定は最大辺を基準にする（全軸がvoxel_size以下になったら確定）
            max_edge = float(child_size.max())

            if max_edge <= voxel_size:
                # flag1: 確定 → SHELLセルとして追加
                cell = Cell(
                    origin=child_min.copy(),
                    size=child_size.copy(),
                    depth=child_depth,
                    state=CellState.SHELL,
                    facet_indices=tuple(hit_indices.tolist()),
                )
                model.add_cell(cell)
            elif child_depth < max_depth:
                # flag2: 再分割
                queue.append((child_min, child_max, hit_indices, child_depth))
            else:
                # 深さ上限に到達 → 強制確定
                logger.warning(
                    "MAX_OCTREE_DEPTH(%d)に到達したため、サイズ閾値未達のまま"
                    "Cellとして強制確定します: min=%s max=%s",
                    max_depth,
                    child_min,
                    child_max,
                )
                cell = Cell(
                    origin=child_min.copy(),
                    size=child_size.copy(),
                    depth=child_depth,
                    state=CellState.SHELL,
                    facet_indices=tuple(hit_indices.tolist()),
                )
                model.add_cell(cell)

    return model


def _split_octants(
    min_corner: np.ndarray, max_corner: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """軸ごとの中点で8分割した子ボックスの(min, max)リストを返す。"""
    mid = (min_corner + max_corner) / 2.0
    octants = []
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                low = np.array(
                    [
                        min_corner[0] if ix == 0 else mid[0],
                        min_corner[1] if iy == 0 else mid[1],
                        min_corner[2] if iz == 0 else mid[2],
                    ]
                )
                high = np.array(
                    [
                        mid[0] if ix == 0 else max_corner[0],
                        mid[1] if iy == 0 else max_corner[1],
                        mid[2] if iz == 0 else max_corner[2],
                    ]
                )
                octants.append((low, high))
    return octants


def _intersecting_facets(
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    facets: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    """候補ファセットのうち、このボックスと交差するものだけのindexを返す（親からの継承で枝刈り）。"""
    if len(candidate_indices) == 0:
        return candidate_indices
    box_center = (min_corner + max_corner) / 2.0
    box_half_size = (max_corner - min_corner) / 2.0
    hits_mask = triangle_box_overlap_batch(box_center, box_half_size, facets[candidate_indices])
    return candidate_indices[hits_mask]
