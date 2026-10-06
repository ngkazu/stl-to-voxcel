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


def point_to_triangle_distance(point: np.ndarray, triangle: np.ndarray) -> float:
    """点から三角形への最短距離を計算する。

    点が三角形の内部に投影される場合は垂直距離、
    そうでない場合は最近接の辺または頂点への距離を返す。
    """
    point = np.asarray(point, dtype=np.float64)
    v0, v1, v2 = np.asarray(triangle, dtype=np.float64)

    # 三角形の辺ベクトル
    edge0 = v1 - v0
    edge1 = v2 - v0
    v0_to_point = point - v0

    # 平面上での重心座標を計算するためのドット積
    d00 = np.dot(edge0, edge0)
    d01 = np.dot(edge0, edge1)
    d11 = np.dot(edge1, edge1)
    d20 = np.dot(v0_to_point, edge0)
    d21 = np.dot(v0_to_point, edge1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < _EPS:
        # 退化三角形（線分または点）→ 3頂点への距離の最小値
        return min(
            np.linalg.norm(point - v0),
            np.linalg.norm(point - v1),
            np.linalg.norm(point - v2),
        )

    # 重心座標 (u, v)
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    w = 1.0 - u - v

    # 点が三角形内部に投影される場合
    if u >= 0 and v >= 0 and w >= 0:
        closest = v0 + u * edge0 + v * edge1
        return float(np.linalg.norm(point - closest))

    # 三角形の外側 → 辺または頂点への距離
    return min(
        _point_to_segment_distance(point, v0, v1),
        _point_to_segment_distance(point, v1, v2),
        _point_to_segment_distance(point, v2, v0),
    )


def _point_to_segment_distance(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    """点から線分への最短距離を計算する。"""
    seg = seg_b - seg_a
    seg_len_sq = np.dot(seg, seg)

    if seg_len_sq < _EPS:
        # 線分が点に退化
        return float(np.linalg.norm(point - seg_a))

    # 線分上の最近接点のパラメータ t ∈ [0, 1]
    t = max(0.0, min(1.0, np.dot(point - seg_a, seg) / seg_len_sq))
    closest = seg_a + t * seg
    return float(np.linalg.norm(point - closest))


def point_to_mesh_distance(point: np.ndarray, facets: np.ndarray) -> float:
    """点からメッシュ（三角形群）への最短距離を計算する。

    facets: shape (N, 3, 3) の三角形配列
    """
    min_dist = float("inf")
    for triangle in facets:
        dist = point_to_triangle_distance(point, triangle)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def points_to_mesh_distance_batch(points: np.ndarray, facets: np.ndarray) -> np.ndarray:
    """複数の点からメッシュへの最短距離を一括計算する。

    points: shape (M, 3)
    facets: shape (N, 3, 3)
    戻り値: shape (M,) の距離配列
    """
    points = np.asarray(points, dtype=np.float64)
    return np.array([point_to_mesh_distance(p, facets) for p in points])


class MeshDistanceCalculator:
    """KD-treeを使った高速なメッシュ距離計算クラス。

    三角形の重心でKD-treeを構築し、近傍三角形のみを候補として
    距離計算を行うことで高速化する。
    """

    def __init__(self, facets: np.ndarray):
        """初期化。

        Args:
            facets: 三角形配列 (N, 3, 3)
        """
        from scipy.spatial import KDTree

        self.facets = np.asarray(facets, dtype=np.float64)
        self.n_facets = len(self.facets)

        # 各三角形の重心を計算
        self.centroids = self.facets.mean(axis=1)  # (N, 3)

        # 各三角形の「半径」（重心から頂点への最大距離）を計算
        diffs = self.facets - self.centroids[:, np.newaxis, :]  # (N, 3, 3)
        self.radii = np.max(np.linalg.norm(diffs, axis=2), axis=1)  # (N,)

        # 中央値の半径を基準にする（外れ値の影響を軽減）
        self.median_radius = float(np.median(self.radii))

        # KD-treeを構築
        self.kdtree = KDTree(self.centroids)

    def distance(self, point: np.ndarray) -> float:
        """点からメッシュへの最短距離を計算する。

        Args:
            point: 点座標 (3,)

        Returns:
            最短距離
        """
        point = np.asarray(point, dtype=np.float64)

        # 段階的に検索半径を広げる
        min_dist = float("inf")

        # 初期検索半径：中央値半径の3倍から開始
        search_radius = self.median_radius * 3

        while True:
            candidate_indices = self.kdtree.query_ball_point(point, search_radius)

            for idx in candidate_indices:
                # 重心からの距離 - 三角形半径 > 現在の最短距離 なら計算不要
                centroid_dist = np.linalg.norm(point - self.centroids[idx])
                if centroid_dist - self.radii[idx] > min_dist:
                    continue

                dist = point_to_triangle_distance(point, self.facets[idx])
                if dist < min_dist:
                    min_dist = dist

            # 検索半径内で見つかった最短距離が、検索半径より十分小さければ終了
            if min_dist < search_radius - self.median_radius * 2:
                break

            # 検索半径を広げる
            search_radius *= 2

            # 全体をカバーしたら終了
            if search_radius > 10000:
                break

        return min_dist

    def distances_batch(self, points: np.ndarray) -> np.ndarray:
        """複数の点からメッシュへの最短距離を一括計算する。

        Args:
            points: 点座標配列 (M, 3)

        Returns:
            距離配列 (M,)
        """
        points = np.asarray(points, dtype=np.float64)
        distances = np.zeros(len(points), dtype=np.float64)

        for i, point in enumerate(points):
            distances[i] = self.distance(point)

        return distances

    def distances_batch_with_hint(
        self,
        points: np.ndarray,
        hint_radius: float,
    ) -> np.ndarray:
        """ヒント半径を使った高速な距離計算。

        expand_distanceなど、期待される距離のヒントがある場合に使用。
        ヒント半径以上離れていることが確認できれば、それ以上計算しない。

        Args:
            points: 点座標配列 (M, 3)
            hint_radius: 期待される距離のヒント（これ以上離れていれば十分）

        Returns:
            距離配列 (M,) - hint_radiusより大きい場合は正確な値ではない可能性あり
        """
        points = np.asarray(points, dtype=np.float64)
        distances = np.full(len(points), float("inf"), dtype=np.float64)

        # 初期検索半径：hint_radius + 中央値半径 * 2（余裕を持たせる）
        initial_search_radius = hint_radius + self.median_radius * 2

        for i, point in enumerate(points):
            min_dist = float("inf")
            search_radius = initial_search_radius

            while True:
                candidate_indices = self.kdtree.query_ball_point(point, search_radius)

                for idx in candidate_indices:
                    # 重心からの距離 - 三角形半径 > 現在の最短距離 なら計算不要
                    centroid_dist = np.linalg.norm(point - self.centroids[idx])
                    if centroid_dist - self.radii[idx] > min_dist:
                        continue

                    dist = point_to_triangle_distance(point, self.facets[idx])
                    if dist < min_dist:
                        min_dist = dist
                        # hint_radiusより小さければ早期終了
                        if min_dist < hint_radius * 0.8:
                            break

                # 十分近い三角形が見つかった、または検索半径内で収束した
                if min_dist < hint_radius or min_dist < search_radius - self.median_radius * 2:
                    break

                # 検索半径を広げる
                search_radius *= 1.5

                # 十分広げたら終了
                if search_radius > initial_search_radius * 10:
                    break

            distances[i] = min_dist

        return distances
