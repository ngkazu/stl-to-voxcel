from __future__ import annotations

import logging

import numpy as np

from core.octree import build_voxels, make_cubic_bbox


def test_single_triangle_in_one_octant_yields_one_voxel() -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([4.0, 4.0, 4.0])
    triangle = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]])
    facets = np.array([triangle])

    voxels = build_voxels(bbox_min, bbox_max, facets, voxel_size=3.0)

    assert len(voxels) == 1
    assert np.allclose(voxels[0].min_corner, [0.0, 0.0, 0.0])
    assert np.allclose(voxels[0].max_corner, [2.0, 2.0, 2.0])


def test_root_always_splits_at_least_once_even_if_voxel_size_is_huge() -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([2.0, 2.0, 2.0])
    triangle = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]])
    facets = np.array([triangle])

    voxels = build_voxels(bbox_min, bbox_max, facets, voxel_size=100.0)

    assert len(voxels) >= 1
    for voxel in voxels:
        edge = voxel.max_corner - voxel.min_corner
        assert np.all(edge < 2.0)  # ルートそのものではなく、必ず分割された子であること


def test_smaller_voxel_size_yields_smaller_voxels() -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([4.0, 4.0, 4.0])
    triangle = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]])
    facets = np.array([triangle])

    coarse = build_voxels(bbox_min, bbox_max, facets, voxel_size=3.0)
    fine = build_voxels(bbox_min, bbox_max, facets, voxel_size=1.5)

    coarse_edge = float((coarse[0].max_corner - coarse[0].min_corner).mean())
    fine_edge = max(float((v.max_corner - v.min_corner).mean()) for v in fine)
    assert fine_edge < coarse_edge


def test_max_depth_forces_voxel_and_logs_warning(caplog) -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([8.0, 8.0, 8.0])
    triangle = np.array([[0.0, 0.0, 0.0], [8.0, 8.0, 0.0], [0.0, 8.0, 8.0]])
    facets = np.array([triangle])

    with caplog.at_level(logging.WARNING):
        voxels = build_voxels(bbox_min, bbox_max, facets, voxel_size=1e-6, max_depth=2)

    assert len(voxels) > 0
    assert any("MAX_OCTREE_DEPTH" in record.message for record in caplog.records)


def test_make_cubic_bbox_expands_to_longest_edge_centered() -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([10.0, 2.0, 4.0])

    cube_min, cube_max = make_cubic_bbox(bbox_min, bbox_max)

    assert np.allclose(cube_max - cube_min, [10.0, 10.0, 10.0])
    # 元のbboxの中心を保ったまま拡張されていること
    original_center = (bbox_min + bbox_max) / 2.0
    cube_center = (cube_min + cube_max) / 2.0
    assert np.allclose(cube_center, original_center)


def test_build_voxels_with_cubic_root_yields_cube_voxels() -> None:
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([8.0, 2.0, 2.0])  # 非立方体のbbox
    triangle = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]])
    facets = np.array([triangle])

    voxels = build_voxels(bbox_min, bbox_max, facets, voxel_size=3.0, cubic_root=True)

    assert len(voxels) >= 1
    for voxel in voxels:
        edge = voxel.max_corner - voxel.min_corner
        assert np.allclose(edge[0], edge[1])
        assert np.allclose(edge[1], edge[2])
