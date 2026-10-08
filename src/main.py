"""CLI エントリポイント。`python src/main.py <stl_path> --voxel-size <値>` で実行する。"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from core.octree import build_voxel_model
from core.solid_fill import fill_solid
from core.stl_loader import load_stl
from core.visualize import show_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="main")
    parser.add_argument("stl_path", help="読み込むSTLファイルのパス")
    parser.add_argument("--voxel-size", type=float, required=True, help="目標ボクセルサイズ")
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="PyVistaによる可視化を行わない",
    )
    parser.add_argument(
        "--cubic-root",
        action="store_true",
        help="ルートのbboxを最大辺に合わせた立方体に拡張してから分割する（軸ごとの分解能偏りを防ぐ）",
    )
    parser.add_argument(
        "--no-solid-fill",
        dest="solid_fill",
        action="store_false",
        help="内部充塡（solid-fill）を行わない（デフォルトは充塡する）",
    )
    parser.add_argument(
        "--cross-section",
        action="store_true",
        help="ドラッグ可能な平面ウィジェットで断面を確認できるようにする",
    )
    args = parser.parse_args(argv)

    # STL読み込み
    facets, bbox_min, bbox_max = load_stl(args.stl_path)
    print(f"facets: {len(facets)}")
    print(f"bbox: {bbox_min.tolist()} - {bbox_max.tolist()}")

    # 8分木でVoxelModel構築（SHELLセル）
    model = build_voxel_model(
        bbox_min, bbox_max, facets, args.voxel_size, cubic_root=args.cubic_root
    )
    print(f"voxel_size: {args.voxel_size}")
    print(f"cells (shell): {model.shell_count}")

    # SHELLセルのサイズ情報を出力（ユニークなサイズ [sx,sy,sz] と個数）
    # cubic_root無しだと非立方体になりうるため、軸ごとに確認できるようにする
    shell_cells = model.get_shell_cells()
    if shell_cells:
        sizes = np.array([model.cell_origin_size(c)[1] for c in shell_cells])
        unique_sizes, counts = np.unique(sizes, axis=0, return_counts=True)
        print(f"cells (shell) sizes: {len(unique_sizes)} unique")
        for size, count in zip(unique_sizes, counts, strict=True):
            print(f"  size=[{size[0]:.4g}, {size[1]:.4g}, {size[2]:.4g}] x {count}")

    # 内部充填
    if args.solid_fill:
        # 充填の格子セルサイズ（SHELLセルの最小サイズ、各軸）
        # cubic_root無しだと非立方体になりうるため np.ndarray で扱う
        min_cell_size = model.get_min_cell_size()
        fill_solid(model, min_cell_size)
        print(f"cells (inside): {model.inside_count}")

    # 統計表示
    print(f"total cells: {model.cell_count}")

    # 可視化
    if not args.no_visualize:
        show_model(model, facets, cross_section=args.cross_section)

    return 0


if __name__ == "__main__":
    sys.exit(main())
