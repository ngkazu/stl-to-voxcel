"""STLファセット群から8分木でVoxelを構築するロジック。"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from core.geometry import triangle_box_overlap_batch
from core.model import ROOT_CODE, Cell, CellState, VoxelModel, code_child

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


def compute_target_depth(span: float, voxel_size: float) -> int:
    """root辺長(span)をvoxel_size以下にするのに必要な8分木のdepthを求める。

    span / 2**depth が voxel_size 以下になる最小のdepthを返す（span<=voxel_sizeなら0）。
    span を voxel_size の「2の累乗倍」に合わせて拡張しておけば、このdepthちょうどで
    セルサイズがvoxel_sizeに厳密一致する（2分割は浮動小数点誤差が乗らないため）。
    """
    if span <= voxel_size:
        return 0
    return math.ceil(math.log2(span / voxel_size))


def plan_octree(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    voxel_size: float,
    max_depth: int = MAX_OCTREE_DEPTH,
    cubic_root: bool = False,
) -> tuple[int, np.ndarray, np.ndarray]:
    """build_voxel_model本体（ファセット走査）を始める前に、必要なdepthと
    ルートbboxを決定する。

    ルートbboxの準備（cubic_root対応・voxel_size整列）とtarget_depthの計算だけを
    行う軽量な処理（ファセット走査は含まない）なので、build_voxel_model本体を
    呼ぶ前に事前確認・表示したい場合はこちらを直接呼べる。

    必要な分割深さが max_depth を超える場合は ValueError を発生させる。

    Returns:
        (target_depth, root_bbox_min, root_bbox_max)
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)

    if cubic_root:
        bbox_min, bbox_max = make_cubic_bbox(bbox_min, bbox_max)

    span = float((bbox_max - bbox_min).mean())
    target_depth = compute_target_depth(span, voxel_size)
    if target_depth > max_depth:
        raise ValueError(
            f"voxel_size({voxel_size})に対して必要な分割深さ({target_depth})が"
            f"MAX_OCTREE_DEPTH({max_depth})を超えています。voxel_sizeを大きくしてください。"
        )

    if cubic_root:
        # セルサイズがvoxel_sizeに厳密一致するよう、一辺長をvoxel_sizeの2の累乗倍に調整する
        # （中心は維持。拡張分はファセットと交差しないため最初の判定で破棄されるだけ）
        center = (bbox_min + bbox_max) / 2.0
        half_size = voxel_size * (2**target_depth) / 2.0
        bbox_min, bbox_max = center - half_size, center + half_size

    return target_depth, bbox_min, bbox_max


def build_voxel_model(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    facets: np.ndarray,
    voxel_size: float,
    max_depth: int = MAX_OCTREE_DEPTH,
    cubic_root: bool = False,
) -> VoxelModel:
    """STL全体のbboxを8分木で再帰分割し、VoxelModelを返す。

    分割すべきdepthとルートbboxは事前に plan_octree で決定し（ノードごとのサイズ判定は
    行わない）、ファセットと交差するセルは必ずそのdepthまで分割される。
    cubic_root=True の場合、確定するSHELLセルのサイズは厳密に voxel_size と一致する。
    """
    target_depth, bbox_min, bbox_max = plan_octree(
        bbox_min, bbox_max, voxel_size, max_depth, cubic_root
    )

    # base_cell_size: ルート（depth=0）のセルサイズ [sx, sy, sz]
    # cubic_root無しだと非立方体（軸ごとに異なる）になりうる。
    base_cell_size = (bbox_max - bbox_min).astype(np.float64)

    model = VoxelModel(
        bbox_min=bbox_min.copy(),
        base_cell_size=base_cell_size.copy(),
        cells={},
    )

    all_facet_indices = np.arange(len(facets))

    if target_depth == 0:
        # ルート自体が既にvoxel_size以下 -> 分割せずルートをそのままセルとして確定する
        hit_indices = _intersecting_facets(bbox_min, bbox_max, facets, all_facet_indices)
        if len(hit_indices) > 0:
            model.add_cell(
                Cell(
                    code=ROOT_CODE,
                    state=CellState.SHELL,
                    facet_indices=tuple(hit_indices.tolist()),
                )
            )
        return model

    # キュー要素: (min_corner, max_corner, candidate_indices, depth, code)
    queue: deque[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = deque()
    queue.append((bbox_min, bbox_max, all_facet_indices, 0, ROOT_CODE))

    while queue:
        min_corner, max_corner, candidate_indices, depth, code = queue.popleft()
        # _split_octants は for ix: for iy: for iz: の順で返すため、
        # enumerate のインデックスがそのまま octant 番号 (ix<<2)|(iy<<1)|iz になる。

        for octant, (child_min, child_max) in enumerate(_split_octants(min_corner, max_corner)):
            hit_indices = _intersecting_facets(child_min, child_max, facets, candidate_indices)
            if len(hit_indices) == 0:
                continue  # 交差なし -> 破棄

            child_depth = depth + 1
            child_code = code_child(code, octant)

            if child_depth >= target_depth:
                # target_depthに到達 -> SHELLセルとして確定
                cell = Cell(
                    code=child_code,
                    state=CellState.SHELL,
                    facet_indices=tuple(hit_indices.tolist()),
                )
                model.add_cell(cell)
            else:
                # まだtarget_depthに届いていない -> 再分割
                queue.append((child_min, child_max, hit_indices, child_depth, child_code))

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
