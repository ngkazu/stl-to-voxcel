"""CLI エントリポイント。`python src/main.py <stl_path> --voxel-size <値>` で実行する。"""

from __future__ import annotations

import argparse
import sys

from core.octree import build_voxels
from core.stl_loader import load_stl
from core.visualize import show_voxels


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
    args = parser.parse_args(argv)

    facets, bbox_min, bbox_max = load_stl(args.stl_path)
    voxels = build_voxels(bbox_min, bbox_max, facets, args.voxel_size, cubic_root=args.cubic_root)

    print(f"facets: {len(facets)}")
    print(f"bbox: {bbox_min.tolist()} - {bbox_max.tolist()}")
    print(f"voxel_size: {args.voxel_size}")
    print(f"voxels (flag1): {len(voxels)}")

    if not args.no_visualize:
        show_voxels(voxels, facets)

    return 0


if __name__ == "__main__":
    sys.exit(main())
