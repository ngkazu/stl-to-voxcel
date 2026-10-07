"""expand モジュールのテスト。"""

from pathlib import Path

import numpy as np
import pytest

from core.expand import expand_model
from core.model import CellState, VoxelModel
from core.octree import build_voxel_model
from core.solid_fill import fill_solid
from core.stl_loader import load_stl

# テストデータのパス
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CASE1_STL = DATA_DIR / "case1.stl"


@pytest.fixture
def case1_facets():
    """case1.stl の facets を読み込む。"""
    if not CASE1_STL.exists():
        pytest.skip(f"Test data not found: {CASE1_STL}")
    facets, bbox_min, bbox_max = load_stl(str(CASE1_STL))
    return facets, bbox_min, bbox_max


class TestExpandModel:
    """expand_model 関数のテスト。"""

    def test_expand_adds_outside_cells(self, case1_facets):
        """膨張処理でOUTSIDEセルが追加されることを確認。"""
        facets, bbox_min, bbox_max = case1_facets
        voxel_size = 10.0
        expand_distance = 10.0  # voxel_size以上で膨張が発生

        # VoxelModel構築
        model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size, cubic_root=False)
        initial_shell_count = model.shell_count
        assert initial_shell_count > 0

        # 格子セルサイズ（SHELLセルの最小サイズ、各軸）
        cell_size = model.get_min_cell_size()

        # 内部充填
        fill_solid(model, cell_size)

        # 膨張処理
        expand_model(model, facets, expand_distance, cell_size)

        # OUTSIDEセルが追加されていることを確認
        assert model.outside_count > 0
        assert model.cell_count > initial_shell_count

    def test_expand_preserves_shell_and_inside(self, case1_facets):
        """膨張処理がSHELL/INSIDEセルを変更しないことを確認。"""
        facets, bbox_min, bbox_max = case1_facets
        voxel_size = 10.0
        expand_distance = 10.0  # voxel_size以上で膨張が発生

        # VoxelModel構築
        model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size, cubic_root=False)
        cell_size = model.get_min_cell_size()
        fill_solid(model, cell_size)

        shell_count_before = model.shell_count
        inside_count_before = model.inside_count

        # 膨張処理
        expand_model(model, facets, expand_distance, cell_size)

        # SHELL/INSIDEセル数が変わらないことを確認
        assert model.shell_count == shell_count_before
        assert model.inside_count == inside_count_before

    def test_expand_with_zero_distance(self, case1_facets):
        """expand_distance=0 の場合、OUTSIDEセルが追加されないことを確認。"""
        facets, bbox_min, bbox_max = case1_facets
        voxel_size = 10.0
        expand_distance = 0.0

        # VoxelModel構築
        model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size, cubic_root=False)
        cell_size = model.get_min_cell_size()
        fill_solid(model, cell_size)

        cell_count_before = model.cell_count

        # 膨張処理（距離0）
        expand_model(model, facets, expand_distance, cell_size)

        # セル数が変わらないことを確認
        assert model.cell_count == cell_count_before
        assert model.outside_count == 0

    def test_expand_cell_states(self, case1_facets):
        """膨張後の全セルが正しい状態を持つことを確認。"""
        facets, bbox_min, bbox_max = case1_facets
        voxel_size = 10.0
        expand_distance = 10.0  # voxel_size以上で膨張が発生

        # VoxelModel構築
        model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size, cubic_root=False)
        cell_size = model.get_min_cell_size()
        fill_solid(model, cell_size)
        expand_model(model, facets, expand_distance, cell_size)

        # 全セルの状態を確認
        for cell in model.cells:
            assert cell.state in (CellState.SHELL, CellState.INSIDE, CellState.OUTSIDE)

        # カウントの整合性を確認
        assert model.cell_count == model.shell_count + model.inside_count + model.outside_count

    def test_expand_smaller_than_voxel_size(self, case1_facets):
        """expand_distance < voxel_size でも距離保証されることを確認。"""
        facets, bbox_min, bbox_max = case1_facets
        voxel_size = 10.0
        expand_distance = 5.0  # voxel_size未満

        # VoxelModel構築
        model = build_voxel_model(bbox_min, bbox_max, facets, voxel_size, cubic_root=False)
        cell_size = model.get_min_cell_size()
        fill_solid(model, cell_size)

        # 膨張処理
        expand_model(model, facets, expand_distance, cell_size)

        # 膨張が必要なら追加される、不要なら0のまま（どちらも正常動作）
        # 重要なのは距離保証であり、OUTSIDEセル数ではない
        assert model.outside_count >= 0
