"""PyVistaを使ったVoxel・元メッシュの可視化。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv

from core.model import Cell, VoxelModel


def show_model(
    model: VoxelModel,
    facets: np.ndarray | None = None,
    cross_section: bool = False,
) -> None:
    """VoxelModelをPyVistaで表示する。

    - SHELLセル: オレンジ
    - INSIDEセル: 緑
    - OUTSIDEセル: 青

    facetsを渡すと元のSTLメッシュを半透明で重ね描画する。
    cross_section=True の場合、ドラッグ可能な平面ウィジェットで断面を確認できるようにする
    （ウィジェットでVoxel単位ごと綺麗に切断されるよう crinkle=True を使う）。
    """
    plotter = pv.Plotter()

    def add(mesh: pv.DataSet, **kwargs: Any) -> pv.Actor:
        if cross_section:
            return plotter.add_mesh_clip_plane(mesh, normal="x", crinkle=True, **kwargs)  # type: ignore[arg-type]
        return plotter.add_mesh(mesh, **kwargs)

    sliders: list[tuple[pv.Actor, str, float]] = []

    shell_cells = model.get_shell_cells()
    inside_cells = model.get_inside_cells()
    outside_cells = model.get_outside_cells()

    if shell_cells:
        actor = add(
            _cells_to_mesh(model, shell_cells), color="orange", opacity=0.5, show_edges=True
        )
        sliders.append((actor, "Shell opacity", 0.5))

    if inside_cells:
        actor = add(
            _cells_to_mesh(model, inside_cells), color="green", opacity=0.6, show_edges=True
        )
        sliders.append((actor, "Inside opacity", 0.6))

    if outside_cells:
        actor = add(
            _cells_to_mesh(model, outside_cells), color="blue", opacity=0.4, show_edges=True
        )
        sliders.append((actor, "Outside opacity", 0.4))

    if facets is not None and len(facets) > 0:
        actor = add(_facets_to_polydata(facets), color="lightblue", opacity=0.3)
        sliders.append((actor, "Mesh opacity", 0.3))

    _add_opacity_sliders(plotter, sliders)

    plotter.show()


def _add_opacity_sliders(plotter: pv.Plotter, sliders: list[tuple[pv.Actor, str, float]]) -> None:
    """各レイヤーのActorごとに透明度スライダーを追加し、ドラッグでリアルタイムに変更できるようにする。"""
    for i, (actor, title, initial_opacity) in enumerate(sliders):
        top = 0.9 - 0.15 * i

        def callback(value: float, actor: pv.Actor = actor) -> None:
            actor.prop.opacity = value

        plotter.add_slider_widget(
            callback,  # type: ignore[arg-type]
            rng=[0.0, 1.0],
            value=initial_opacity,
            title=title,
            pointa=(0.025, top),
            pointb=(0.31, top),
        )


def _cells_to_mesh(model: VoxelModel, cells: list[Cell]) -> pv.UnstructuredGrid:
    """Cellごとの直方体を1つのUnstructuredGridに統合する。

    pv.Box を大量生成すると遅いため、全セルの8頂点と六面体(hexahedron)接続を
    numpy配列で直接構築し、一括で UnstructuredGrid を作る。
    """
    n = len(cells)
    if n == 0:
        return pv.UnstructuredGrid()

    # 各セルの origin / size を取得（shrink後は cell_face_positions が実際の位置を返す）
    origins = np.empty((n, 3), dtype=np.float64)
    sizes = np.empty((n, 3), dtype=np.float64)
    for i, cell in enumerate(cells):
        face_positions = model.cell_face_positions(cell)
        origin = face_positions[0::2]  # [min_x, min_y, min_z]
        origins[i] = origin
        sizes[i] = face_positions[1::2] - origin  # [max_x-min_x, max_y-min_y, max_z-min_z]

    # 各セルの8頂点オフセット（単位立方体）: VTK HEXAHEDRON の頂点順
    # 0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0) 4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
    unit = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )

    # 全頂点を計算: (n, 8, 3) -> (n*8, 3)
    points = origins[:, None, :] + unit[None, :, :] * sizes[:, None, :]
    points = points.reshape(-1, 3)

    # セル接続配列: 各セル [8, p0, p1, ..., p7]
    base = np.arange(n, dtype=np.int64) * 8
    connectivity = np.empty((n, 9), dtype=np.int64)
    connectivity[:, 0] = 8  # 頂点数
    connectivity[:, 1:] = base[:, None] + np.arange(8, dtype=np.int64)[None, :]
    cells_array = connectivity.ravel()

    from vtkmodules.util.vtkConstants import VTK_HEXAHEDRON  # noqa: PLC0415

    cell_types = np.full(n, VTK_HEXAHEDRON, dtype=np.uint8)

    return pv.UnstructuredGrid(cells_array, cell_types, points)


def _facets_to_polydata(facets: np.ndarray) -> pv.PolyData:
    """(N,3,3)形式のファセット配列をPyVistaのPolyDataに変換する。"""
    n_facets = facets.shape[0]
    points = facets.reshape(-1, 3)
    indices = np.arange(n_facets * 3).reshape(n_facets, 3)
    faces = np.hstack([np.full((n_facets, 1), 3), indices]).astype(np.int64).ravel()
    return pv.PolyData(points, faces)
