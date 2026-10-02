from __future__ import annotations

import numpy as np

from core.geometry import triangle_box_overlap, triangle_box_overlap_batch

BOX_CENTER = np.array([0.0, 0.0, 0.0])
BOX_HALF_SIZE = np.array([1.0, 1.0, 1.0])


def test_triangle_fully_inside_box() -> None:
    triangle = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]])
    assert triangle_box_overlap(BOX_CENTER, BOX_HALF_SIZE, triangle) is True


def test_triangle_fully_outside_box() -> None:
    triangle = np.array([[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0]])
    assert triangle_box_overlap(BOX_CENTER, BOX_HALF_SIZE, triangle) is False


def test_triangle_crossing_boundary() -> None:
    triangle = np.array([[0.5, 0.5, 0.0], [2.0, 0.5, 0.0], [0.5, 2.0, 0.0]])
    assert triangle_box_overlap(BOX_CENTER, BOX_HALF_SIZE, triangle) is True


def test_triangle_touching_only_at_vertex() -> None:
    triangle = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
    assert triangle_box_overlap(BOX_CENTER, BOX_HALF_SIZE, triangle) is True


def test_degenerate_zero_area_triangle_does_not_crash() -> None:
    triangle = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert triangle_box_overlap(BOX_CENTER, BOX_HALF_SIZE, triangle) is True

    triangle_outside = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]])
    assert triangle_box_overlap(BOX_CENTER, BOX_HALF_SIZE, triangle_outside) is False


def test_triangle_box_overlap_batch_matches_scalar_results() -> None:
    triangles = np.array(
        [
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],  # 内包
            [[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0]],  # 外部
            [[0.5, 0.5, 0.0], [2.0, 0.5, 0.0], [0.5, 2.0, 0.0]],  # 境界またぎ
            [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]],  # 頂点のみ接触
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # 退化(内側)
        ]
    )
    expected = np.array([triangle_box_overlap(BOX_CENTER, BOX_HALF_SIZE, t) for t in triangles])
    actual = triangle_box_overlap_batch(BOX_CENTER, BOX_HALF_SIZE, triangles)
    assert np.array_equal(actual, expected)


def test_triangle_box_overlap_batch_empty_input() -> None:
    empty = np.zeros((0, 3, 3))
    result = triangle_box_overlap_batch(BOX_CENTER, BOX_HALF_SIZE, empty)
    assert result.shape == (0,)
