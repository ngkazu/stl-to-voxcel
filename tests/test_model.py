"""model モジュール（locational code）のテスト。"""

from __future__ import annotations

import numpy as np

from core.model import (
    ROOT_CODE,
    Cell,
    CellState,
    VoxelModel,
    code_child,
    code_depth,
    code_from_grid_index,
    code_parent,
    code_to_grid_index,
    code_to_origin_size,
    octant_from_bits,
)


class TestLocationalCodeBasics:
    """locational code の基本演算のテスト。"""

    def test_root_depth_is_zero(self):
        assert code_depth(ROOT_CODE) == 0

    def test_depth_increases_with_children(self):
        c1 = code_child(ROOT_CODE, 0)
        assert code_depth(c1) == 1
        c2 = code_child(c1, 5)
        assert code_depth(c2) == 2
        c3 = code_child(c2, 7)
        assert code_depth(c3) == 3

    def test_parent_of_child_is_original(self):
        for octant in range(8):
            c = code_child(ROOT_CODE, octant)
            assert code_parent(c) == ROOT_CODE

    def test_parent_chain(self):
        c = ROOT_CODE
        for octant in [1, 3, 7, 0, 5]:
            c = code_child(c, octant)
        # 5回親を辿るとルートに戻る
        for _ in range(5):
            c = code_parent(c)
        assert c == ROOT_CODE

    def test_parent_of_root_is_root(self):
        assert code_parent(ROOT_CODE) == ROOT_CODE

    def test_octant_from_bits(self):
        assert octant_from_bits(0, 0, 0) == 0
        assert octant_from_bits(0, 0, 1) == 1
        assert octant_from_bits(0, 1, 0) == 2
        assert octant_from_bits(1, 0, 0) == 4
        assert octant_from_bits(1, 1, 1) == 7

    def test_sentinel_distinguishes_leading_zeros(self):
        # depth1の子0 と depth2の子0,0 は異なるcodeになる（番兵ビットの効果）
        c_depth1 = code_child(ROOT_CODE, 0)
        c_depth2 = code_child(c_depth1, 0)
        assert c_depth1 != c_depth2
        assert code_depth(c_depth1) == 1
        assert code_depth(c_depth2) == 2


class TestGridIndexRoundtrip:
    """グリッドインデックス ↔ code の往復変換テスト。"""

    def test_roundtrip_depth1(self):
        depth = 1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    code = code_from_grid_index((i, j, k), depth)
                    assert code_depth(code) == depth
                    assert code_to_grid_index(code) == (i, j, k)

    def test_roundtrip_depth3(self):
        depth = 3
        dim = 2**depth
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    code = code_from_grid_index((i, j, k), depth)
                    assert code_to_grid_index(code) == (i, j, k)

    def test_roundtrip_depth5_sample(self):
        depth = 5
        dim = 2**depth
        # 全点は多いのでサンプリング
        for i, j, k in [(0, 0, 0), (dim - 1, dim - 1, dim - 1), (3, 17, 28), (31, 0, 15)]:
            code = code_from_grid_index((i, j, k), depth)
            assert code_to_grid_index(code) == (i, j, k)


class TestCodeToOriginSize:
    """code → origin/size の復元テスト。"""

    def test_root_origin_size(self):
        bbox_min = np.array([0.0, 0.0, 0.0])
        base = np.array([8.0, 8.0, 8.0])
        origin, size = code_to_origin_size(ROOT_CODE, bbox_min, base)
        assert np.allclose(origin, [0.0, 0.0, 0.0])
        assert np.allclose(size, [8.0, 8.0, 8.0])

    def test_child_origin_size_cubic(self):
        bbox_min = np.array([0.0, 0.0, 0.0])
        base = np.array([8.0, 8.0, 8.0])
        # 子0 (0,0,0) は原点側、サイズ半分
        c0 = code_child(ROOT_CODE, 0)
        origin, size = code_to_origin_size(c0, bbox_min, base)
        assert np.allclose(origin, [0.0, 0.0, 0.0])
        assert np.allclose(size, [4.0, 4.0, 4.0])
        # 子7 (1,1,1) は反対側
        c7 = code_child(ROOT_CODE, 7)
        origin, size = code_to_origin_size(c7, bbox_min, base)
        assert np.allclose(origin, [4.0, 4.0, 4.0])
        assert np.allclose(size, [4.0, 4.0, 4.0])

    def test_non_cubic_base(self):
        # 非立方体bbox
        bbox_min = np.array([0.0, 0.0, 0.0])
        base = np.array([8.0, 4.0, 2.0])
        c7 = code_child(ROOT_CODE, 7)  # (1,1,1)側
        origin, size = code_to_origin_size(c7, bbox_min, base)
        assert np.allclose(size, [4.0, 2.0, 1.0])
        assert np.allclose(origin, [4.0, 2.0, 1.0])

    def test_grid_index_matches_origin(self):
        # グリッドインデックスから作ったcodeのoriginが、インデックス*cellsizeと一致
        bbox_min = np.array([0.0, 0.0, 0.0])
        base = np.array([8.0, 8.0, 8.0])
        depth = 3
        cell_size = base / (2**depth)  # [1,1,1]
        for i, j, k in [(0, 0, 0), (7, 7, 7), (3, 5, 1)]:
            code = code_from_grid_index((i, j, k), depth)
            origin, size = code_to_origin_size(code, bbox_min, base)
            assert np.allclose(origin, bbox_min + np.array([i, j, k]) * cell_size)
            assert np.allclose(size, cell_size)


class TestVoxelModel:
    """VoxelModel の基本動作テスト。"""

    def test_add_and_count(self):
        model = VoxelModel(
            bbox_min=np.array([0.0, 0.0, 0.0]),
            base_cell_size=np.array([8.0, 8.0, 8.0]),
        )
        model.add_cell(Cell(code=code_child(ROOT_CODE, 0), state=CellState.SHELL))
        model.add_cell(Cell(code=code_child(ROOT_CODE, 1), state=CellState.INSIDE))
        assert model.shell_count == 1
        assert model.inside_count == 1
        assert model.cell_count == 2

    def test_duplicate_code_overwrites(self):
        model = VoxelModel(
            bbox_min=np.array([0.0, 0.0, 0.0]),
            base_cell_size=np.array([8.0, 8.0, 8.0]),
        )
        code = code_child(ROOT_CODE, 0)
        model.add_cell(Cell(code=code, state=CellState.SHELL))
        model.add_cell(Cell(code=code, state=CellState.INSIDE))
        # 同じcodeは上書き
        assert model.cell_count == 1
        assert model.inside_count == 1
        assert model.shell_count == 0

    def test_min_cell_size(self):
        model = VoxelModel(
            bbox_min=np.array([0.0, 0.0, 0.0]),
            base_cell_size=np.array([8.0, 8.0, 8.0]),
        )
        # depth2のセルを追加 → サイズは 8/4 = 2
        c = code_child(code_child(ROOT_CODE, 0), 0)
        model.add_cell(Cell(code=c, state=CellState.SHELL))
        assert np.allclose(model.get_min_cell_size(), [2.0, 2.0, 2.0])
