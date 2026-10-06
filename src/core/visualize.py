"""PyVistaを使ったVoxel・元メッシュの可視化。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv

from core.octree import Voxel


def show_voxels(
    voxels: list[Voxel],
    facets: np.ndarray | None = None,
    fill_voxels: list[Voxel] | None = None,
    cross_section: bool = False,
) -> None:
    """確定したVoxel群をPyVistaで表示する。

    facetsを渡すと元のSTLメッシュを半透明で重ね描画する。
    fill_voxelsを渡すと、ソリッド充塡で得られた内部セルをシェルとは別の色で重ね描画する。
    cross_section=True の場合、ドラッグ可能な平面ウィジェットで断面を確認できるようにする
    （ウィジェットでVoxel単位ごと綺麗に切断されるよう crinkle=True を使う）。
    """
    plotter = pv.Plotter()

    def add(mesh: pv.DataSet, **kwargs: Any) -> pv.Actor:
        if cross_section:
            # pyvistaの内部フォワーディング実装の型スタブの都合でmypyが誤検知するため無視する。
            return plotter.add_mesh_clip_plane(mesh, normal="x", crinkle=True, **kwargs)  # type: ignore[arg-type]
        return plotter.add_mesh(mesh, **kwargs)

    sliders: list[tuple[pv.Actor, str, float]] = []

    if voxels:
        actor = add(_voxels_to_mesh(voxels), color="orange", opacity=0.5, show_edges=True)
        sliders.append((actor, "Shell opacity", 0.5))

    if fill_voxels:
        actor = add(_voxels_to_mesh(fill_voxels), color="green", opacity=0.6, show_edges=True)
        sliders.append((actor, "Fill opacity", 0.6))

    if facets is not None and len(facets) > 0:
        actor = add(_facets_to_polydata(facets), color="lightblue", opacity=0.3)
        sliders.append((actor, "Mesh opacity", 0.3))

    _add_opacity_sliders(plotter, sliders)

    plotter.show()


def _add_opacity_sliders(plotter: pv.Plotter, sliders: list[tuple[pv.Actor, str, float]]) -> None:
    """各レイヤーのActorごとに透明度スライダーを追加し、ドラッグでリアルタイムに変更できるようにする。"""
    for i, (actor, title, initial_opacity) in enumerate(sliders):
        top = 0.9 - 0.15 * i  # スライダーが重ならないよう縦方向にずらして配置する

        def callback(value: float, actor: pv.Actor = actor) -> None:
            actor.prop.opacity = value

        plotter.add_slider_widget(
            callback,  # type: ignore[arg-type]  # pyvistaの型スタブの都合でmypyが誤検知する
            rng=[0.0, 1.0],
            value=initial_opacity,
            title=title,
            pointa=(0.025, top),
            pointb=(0.31, top),
        )


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
