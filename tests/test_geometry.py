from __future__ import annotations

import numpy as np

from core.geometry import (
    clip_triangle_to_rect,
    sweep_distance_to_facets,
    triangle_box_overlap,
    triangle_box_overlap_batch,
)

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


def test_clip_triangle_fully_inside_rect_unchanged() -> None:
    triangle = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    rect = (-10.0, 10.0, -10.0, 10.0)

    clipped = clip_triangle_to_rect(triangle, axis_a=0, axis_b=1, rect=rect)

    assert clipped.shape[0] == 3
    assert np.allclose(sorted(clipped.tolist()), sorted(triangle.tolist()))


def test_clip_triangle_fully_outside_rect_is_empty() -> None:
    triangle = np.array([[20.0, 20.0, 0.0], [21.0, 20.0, 0.0], [20.0, 21.0, 0.0]])
    rect = (-10.0, 10.0, -10.0, 10.0)

    clipped = clip_triangle_to_rect(triangle, axis_a=0, axis_b=1, rect=rect)

    assert clipped.shape == (0, 3)


def test_clip_triangle_corner_cut_produces_quadrilateral() -> None:
    # 直角三角形の1つの頂点だけが rect の外側(x>2)にはみ出すケース
    triangle = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    rect = (-10.0, 2.0, -10.0, 10.0)

    clipped = clip_triangle_to_rect(triangle, axis_a=0, axis_b=1, rect=rect)

    assert clipped.shape[0] == 4
    assert np.all(clipped[:, 0] <= 2.0 + 1e-9)


def test_sweep_distance_hits_facet_in_middle_of_rect() -> None:
    # 矩形の四隅のどのレイとも重ならない、中央だけの小さいファセット
    # （4頂点サンプリングでは検出できないが、面積ベースのクリッピングなら検出できる）
    triangle = np.array([[-0.5, -0.5, 3.0], [0.5, -0.5, 3.0], [0.0, 0.5, 3.0]])
    facets = np.array([triangle])

    distance = sweep_distance_to_facets(
        face_coord=0.0,
        rect=(-5.0, 5.0, -5.0, 5.0),
        axis=2,
        direction=1,
        facets=facets,
    )

    assert distance is not None
    assert np.isclose(distance, 3.0)


def test_sweep_distance_no_overlap_returns_none() -> None:
    triangle = np.array([[10.0, 10.0, 3.0], [11.0, 10.0, 3.0], [10.0, 11.0, 3.0]])
    facets = np.array([triangle])

    distance = sweep_distance_to_facets(
        face_coord=0.0,
        rect=(-5.0, 5.0, -5.0, 5.0),
        axis=2,
        direction=1,
        facets=facets,
    )

    assert distance is None


def test_sweep_distance_picks_nearest_of_multiple_facets() -> None:
    near = np.array([[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0], [0.0, 0.5, 2.0]])
    far = np.array([[-0.5, -0.5, 5.0], [0.5, -0.5, 5.0], [0.0, 0.5, 5.0]])
    facets = np.array([far, near])

    distance = sweep_distance_to_facets(
        face_coord=0.0,
        rect=(-5.0, 5.0, -5.0, 5.0),
        axis=2,
        direction=1,
        facets=facets,
    )

    assert distance is not None
    assert np.isclose(distance, 2.0)


def test_sweep_distance_direction_negative() -> None:
    # -方向へのスイープ（faceが上側にあり、下方向のSTLへ移動する想定）
    triangle = np.array([[-0.5, -0.5, -2.0], [0.5, -0.5, -2.0], [0.0, 0.5, -2.0]])
    facets = np.array([triangle])

    distance = sweep_distance_to_facets(
        face_coord=0.0,
        rect=(-5.0, 5.0, -5.0, 5.0),
        axis=2,
        direction=-1,
        facets=facets,
    )

    assert distance is not None
    assert np.isclose(distance, 2.0)
