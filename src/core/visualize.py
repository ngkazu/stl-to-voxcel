"""PyVistaを使ったVoxel・元メッシュの可視化。"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from core.octree import Voxel


def show_voxels(voxels: list[Voxel], facets: np.ndarray | None = None) -> None:
    """確定したVoxel群をPyVistaで表示する。facetsを渡すと元のSTLメッシュを半透明で重ね描画する。"""
    plotter = pv.Plotter()

    if voxels:
        plotter.add_mesh(_voxels_to_mesh(voxels), color="orange", opacity=0.5, show_edges=True)

    if facets is not None and len(facets) > 0:
        plotter.add_mesh(_facets_to_polydata(facets), color="lightblue", opacity=0.3)

    plotter.show()


def _voxels_to_mesh(voxels: list[Voxel]) -> pv.UnstructuredGrid:
    """Voxelごとのboxを1つのメッシュに統合する（add_meshをVoxel数回呼ぶと描画が極端に遅くなるため）。"""
    boxes = [
        pv.Box(
            bounds=(
                voxel.min_corner[0],
                voxel.max_corner[0],
                voxel.min_corner[1],
                voxel.max_corner[1],
                voxel.min_corner[2],
                voxel.max_corner[2],
            )
        )
        for voxel in voxels
    ]
    return pv.MultiBlock(boxes).combine()


def _facets_to_polydata(facets: np.ndarray) -> pv.PolyData:
    """(N,3,3)形式のファセット配列をPyVistaのPolyDataに変換する。"""
    n_facets = facets.shape[0]
    points = facets.reshape(-1, 3)
    indices = np.arange(n_facets * 3).reshape(n_facets, 3)
    faces = np.hstack([np.full((n_facets, 1), 3), indices]).astype(np.int64).ravel()
    return pv.PolyData(points, faces)
