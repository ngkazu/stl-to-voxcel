"""三角形とAABB（軸平行境界ボックス）の交差判定。"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def triangle_box_overlap(
    box_center: np.ndarray,
    box_half_size: np.ndarray,
    triangle: np.ndarray,
) -> bool:
    """三角形(triangle, shape=(3,3))がAABB(box_center ± box_half_size)と交差するか判定する。

    Tomas Akenine-Möller の "Fast 3D Triangle-Box Overlap Test"
    （分離軸定理、13本の分離軸）に基づく。
    """
    vertices = np.asarray(triangle, dtype=np.float64) - box_center

    # 軸1-3: boxの3軸（x, y, z）
    for axis_index in range(3):
        proj = vertices[:, axis_index]
        half = box_half_size[axis_index]
        if proj.min() > half or proj.max() < -half:
            return False

    edge0 = vertices[1] - vertices[0]
    edge1 = vertices[2] - vertices[1]
    edge2 = vertices[0] - vertices[2]

    # 軸4: 三角形の法線（平面-box交差判定）
    normal = np.cross(edge0, edge1)
    distance = float(np.dot(normal, vertices[0]))
    radius = float(box_half_size @ np.abs(normal))
    if abs(distance) > radius:
        return False

    # 軸5-13: boxの3軸 × 三角形の3辺 の外積
    box_axes = np.eye(3)
    for edge in (edge0, edge1, edge2):
        for box_axis in box_axes:
            axis = np.cross(box_axis, edge)
            if np.dot(axis, axis) < _EPS:
                continue  # 辺とbox軸がほぼ平行で有効な分離軸にならない
            proj = vertices @ axis
            axis_radius = box_half_size @ np.abs(axis)
            if proj.min() > axis_radius or proj.max() < -axis_radius:
                return False

    return True


def triangle_box_overlap_batch(
    box_center: np.ndarray,
    box_half_size: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """複数の三角形(triangles, shape=(M,3,3))について同一AABBとの交差有無を一括判定する。

    triangle_box_overlap と同じSAT判定を、Pythonループなしでnumpyブロードキャストにより
    まとめて計算する（戻り値は shape (M,) のbool配列）。
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    verts = triangles - box_center
    separated = np.zeros(triangles.shape[0], dtype=bool)

    # 軸1-3: boxの3軸（x, y, z）
    for axis_index in range(3):
        proj = verts[:, :, axis_index]
        half = box_half_size[axis_index]
        separated |= (proj.min(axis=1) > half) | (proj.max(axis=1) < -half)

    edge0 = verts[:, 1] - verts[:, 0]
    edge1 = verts[:, 2] - verts[:, 1]
    edge2 = verts[:, 0] - verts[:, 2]

    # 軸4: 三角形の法線（平面-box交差判定）
    normal = np.cross(edge0, edge1)
    distance = np.einsum("ij,ij->i", normal, verts[:, 0])
    radius = np.abs(normal) @ box_half_size
    separated |= np.abs(distance) > radius

    # 軸5-13: boxの3軸 × 三角形の3辺 の外積
    box_axes = np.eye(3)
    for edge in (edge0, edge1, edge2):
        for box_axis in box_axes:
            axis = np.cross(box_axis, edge)
            axis_len_sq = np.einsum("ij,ij->i", axis, axis)
            proj = np.einsum("mkj,mj->mk", verts, axis)
            axis_radius = np.abs(axis) @ box_half_size
            sep_axis = (proj.min(axis=1) > axis_radius) | (proj.max(axis=1) < -axis_radius)
            separated |= sep_axis & (axis_len_sq >= _EPS)

    return ~separated
