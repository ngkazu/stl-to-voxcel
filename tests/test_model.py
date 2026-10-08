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
    neighbor_code,
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


class TestNeighborCode:
    """neighbor_code（同一depthでの隣接セル探索）のテスト。"""

    def test_neighbor_code_same_depth(self):
        depth = 2
        code = code_from_grid_index((1, 1, 1), depth)

        n = neighbor_code(code, axis=0, direction=1)
        assert n is not None
        assert code_to_grid_index(n) == (2, 1, 1)
        assert code_depth(n) == depth

        n = neighbor_code(code, axis=0, direction=-1)
        assert code_to_grid_index(n) == (0, 1, 1)

        n = neighbor_code(code, axis=1, direction=1)
        assert code_to_grid_index(n) == (1, 2, 1)

        n = neighbor_code(code, axis=2, direction=-1)
        assert code_to_grid_index(n) == (1, 1, 0)

    def test_neighbor_code_boundary_returns_none(self):
        depth = 2
        dim = 2**depth
        code_min = code_from_grid_index((0, 0, 0), depth)
        assert neighbor_code(code_min, axis=0, direction=-1) is None
        assert neighbor_code(code_min, axis=1, direction=-1) is None
        assert neighbor_code(code_min, axis=2, direction=-1) is None

        code_max = code_from_grid_index((dim - 1, dim - 1, dim - 1), depth)
        assert neighbor_code(code_max, axis=0, direction=1) is None
        assert neighbor_code(code_max, axis=1, direction=1) is None
        assert neighbor_code(code_max, axis=2, direction=1) is None

    def test_neighbor_code_root_has_no_neighbor(self):
        # ルート(depth=0)は単一セルなので、どの方向も境界外
        assert neighbor_code(ROOT_CODE, axis=0, direction=1) is None
        assert neighbor_code(ROOT_CODE, axis=0, direction=-1) is None


class TestGetNeighborState:
    """VoxelModel.get_neighbor_state のテスト。"""

    def _empty_model(self) -> VoxelModel:
        return VoxelModel(
            bbox_min=np.array([0.0, 0.0, 0.0]),
            base_cell_size=np.array([8.0, 8.0, 8.0]),
        )

    def test_same_depth_neighbor_found(self):
        model = self._empty_model()
        depth = 2
        cell_a = Cell(code=code_from_grid_index((1, 1, 1), depth), state=CellState.SHELL)
        model.add_cell(cell_a)
        model.add_cell(Cell(code=code_from_grid_index((2, 1, 1), depth), state=CellState.INSIDE))

        assert model.get_neighbor_state(cell_a, axis=0, direction=1) == CellState.INSIDE

    def test_boundary_returns_none(self):
        model = self._empty_model()
        depth = 2
        cell_a = Cell(code=code_from_grid_index((0, 0, 0), depth), state=CellState.SHELL)
        model.add_cell(cell_a)

        assert model.get_neighbor_state(cell_a, axis=0, direction=-1) is None

    def test_unregistered_neighbor_with_no_ancestor_returns_none(self):
        model = self._empty_model()
        depth = 2
        cell_a = Cell(code=code_from_grid_index((1, 1, 1), depth), state=CellState.SHELL)
        model.add_cell(cell_a)

        # 隣接(2,1,1)もその祖先も未登録 -> EMPTY扱いでNone
        assert model.get_neighbor_state(cell_a, axis=0, direction=1) is None

    def test_falls_back_to_parent_when_same_depth_missing(self):
        model = self._empty_model()
        depth2 = 2
        cell_a = Cell(code=code_from_grid_index((1, 1, 1), depth2), state=CellState.SHELL)
        model.add_cell(cell_a)

        # 隣接(2,1,1)のdepth2セルは登録せず、その親(depth1)だけ登録する
        neighbor_d2 = code_from_grid_index((2, 1, 1), depth2)
        model.add_cell(Cell(code=code_parent(neighbor_d2), state=CellState.OUTSIDE))

        assert model.get_neighbor_state(cell_a, axis=0, direction=1) == CellState.OUTSIDE


class TestCellFacePositions:
    """VoxelModel.cell_face_positions のテスト。"""

    def test_default_matches_origin_size(self):
        model = VoxelModel(
            bbox_min=np.array([0.0, 0.0, 0.0]),
            base_cell_size=np.array([8.0, 8.0, 8.0]),
        )
        cell = Cell(code=code_child(ROOT_CODE, 0), state=CellState.SHELL)
        model.add_cell(cell)

        positions = model.cell_face_positions(cell)
        origin, size = model.cell_origin_size(cell)
        expected = np.array(
            [
                origin[0],
                origin[0] + size[0],
                origin[1],
                origin[1] + size[1],
                origin[2],
                origin[2] + size[2],
            ]
        )
        assert np.allclose(positions, expected)

    def test_explicit_face_positions_override_default(self):
        model = VoxelModel(
            bbox_min=np.array([0.0, 0.0, 0.0]),
            base_cell_size=np.array([8.0, 8.0, 8.0]),
        )
        custom = np.array([0.0, 3.5, 0.0, 4.0, 0.0, 4.0])
        cell = Cell(code=code_child(ROOT_CODE, 0), state=CellState.SHELL, face_positions=custom)
        model.add_cell(cell)

        assert np.allclose(model.cell_face_positions(cell), custom)
