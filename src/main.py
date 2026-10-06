"""CLI エントリポイント。`python src/main.py <stl_path> --voxel-size <値>` で実行する。"""

from __future__ import annotations

import argparse
import sys

from core.expand import expand_and_validate, occupancy_to_voxels
from core.octree import build_voxels, make_cubic_bbox
from core.solid_fill import fill_solid, rasterize_voxels
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
    parser.add_argument(
        "--solid-fill",
        action="store_true",
        help="外側からのフラッドフィル+パリティ判定で内部を充塡する（入れ子の空洞は空気のまま残す）",
    )
    parser.add_argument(
        "--cross-section",
        action="store_true",
        help="ドラッグ可能な平面ウィジェットで断面を確認できるようにする",
    )
    parser.add_argument(
        "--expand",
        type=float,
        default=None,
        help="Voxel群を外側に膨張させ、STL表面から指定距離以上離れることを保証する",
    )
    args = parser.parse_args(argv)

    facets, bbox_min, bbox_max = load_stl(args.stl_path)
    print(f"facets: {len(facets)}")
    print(f"bbox: {bbox_min.tolist()} - {bbox_max.tolist()}")

    voxels = build_voxels(bbox_min, bbox_max, facets, args.voxel_size, cubic_root=args.cubic_root)

    print(f"voxel_size: {args.voxel_size}")
    print(f"voxels (shell): {len(voxels)}")

    # 内部充填用のbbox
    fill_bbox_min, fill_bbox_max = bbox_min, bbox_max
    if args.cubic_root:
        fill_bbox_min, fill_bbox_max = make_cubic_bbox(bbox_min, bbox_max)

    fill_voxels = None
    expanded_voxels = None

    if args.solid_fill or args.expand is not None:
        # solid_fillまたはexpandが指定された場合は内部充填を行う
        fill_voxels = fill_solid(voxels, fill_bbox_min, fill_bbox_max, args.voxel_size)
        print(f"voxels (solid fill): {len(fill_voxels)}")

        if args.expand is not None:
            # 膨張処理
            print(f"expand distance: {args.expand}")

            # Shell + Fill を結合してoccupancyグリッドを作成
            all_voxels = voxels + fill_voxels
            occupancy, origin = rasterize_voxels(
                all_voxels, fill_bbox_min, fill_bbox_max, args.voxel_size
            )
            print(f"occupancy grid shape: {occupancy.shape}")

            # 膨張 + 距離検証
            validated_occupancy = expand_and_validate(
                all_voxels, occupancy, origin, args.voxel_size, facets, args.expand
            )

            # Voxelリストに変換
            expanded_voxels = occupancy_to_voxels(validated_occupancy, origin, args.voxel_size)
            print(f"voxels (expanded & validated): {len(expanded_voxels)}")

    if not args.no_visualize:
        if expanded_voxels is not None:
            # 膨張結果を表示（元のshell/fillは非表示）
            show_voxels([], facets, fill_voxels=expanded_voxels, cross_section=args.cross_section)
        else:
            show_voxels(voxels, facets, fill_voxels=fill_voxels, cross_section=args.cross_section)

    return 0


if __name__ == "__main__":
    sys.exit(main())
