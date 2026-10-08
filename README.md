# stl-to-voxcel

Convert an STL mesh into a sparse set of axis-aligned voxel boxes ("Voxel") using an
octree (8-way) subdivision, and visualize the result with PyVista.

## What it does

1. Load an STL file and compute its bounding box.
2. Recursively split the bounding box into 8 octants (octree).
3. For each octant, test which STL facets overlap it (SAT-based triangle-box overlap test).
   - No overlapping facets -> discard the octant.
   - Overlapping facets and the octant's average edge length is below `--voxel-size`
     -> accept it as a final Voxel.
   - Overlapping facets but still above `--voxel-size` -> split it further.
4. A safety limit (`MAX_OCTREE_DEPTH` in `src/core/octree.py`) forces a leaf to be
   accepted even if the size threshold was not reached, to avoid runaway recursion
   from a too-small `--voxel-size`.
5. Show the accepted Voxels (and optionally the original mesh) in a PyVista window.

This produces a *shell* voxelization (only voxels touching the STL surface). By default
the interior is also filled (**solid fill**): the shell is rasterized onto a
uniform grid, empty cells are grouped into 6-connected regions, and a region-adjacency
parity walk (even/odd number of shell layers crossed from the outside) decides which
regions are solid material (filled) versus nested interior cavities (left empty). Pass
`--no-solid-fill` to produce a shell-only voxelization. See
`src/core/solid_fill.py` for details.

The STL format itself carries no unit information — `--voxel-size` is
interpreted in whatever unit the STL coordinates use.

## Project layout

```
stl-to-voxcel/
  src/
    main.py              # CLI entry point
    core/
      model.py            # Data structures: Cell, CellState, VoxelModel (locational code)
      stl_loader.py       # STL loading (numpy-stl) + degenerate-facet filtering
      geometry.py         # Triangle-box overlap test, point-to-mesh distance (KD-tree), sweep distance
      octree.py           # Octree subdivision -> VoxelModel (SHELL cells)
      solid_fill.py       # Interior fill via flood-fill + shell-crossing parity (INSIDE cells)
      shrink.py           # Shrink SHELL outer faces to the STL surface (greedy-merged faces + sweep distance)
      export.py           # Export SHELL+INSIDE boundary faces as an STL file
      visualize.py        # PyVista rendering (state-based colors, cross-section clip)
  tests/                  # pytest unit tests
  data/                   # Sample/test STL files
  docs/, memos/, references/  # Design notes, planning memos, reference material
  requirements.txt        # numpy, numpy-stl, pyvista, scipy
  requirements-dev.txt    # pytest, ruff, mypy
```

## Setup

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

## Usage

```powershell
python src\main.py <path-to-file.stl> --voxel-size <size> [options]
```

Options:

| Flag | Effect |
|---|---|
| `--cubic-root` | Expand the root bounding box to a cube (longest edge, same center) before subdividing, so every Voxel is a cube instead of inheriting the STL's aspect ratio. |
| `--no-solid-fill` | Produce a shell-only voxelization (skip the interior fill). By default the interior is filled; nested cavities are correctly left empty (see "What it does" above). |
| `--shrink` | Shrink SHELL outer faces inward to touch the STL surface (greedy-merged faces + area-based sweep distance). See `src/core/shrink.py`. |
| `--export-stl PATH` | Write the SHELL+INSIDE boundary faces (faces touching EMPTY/OUTSIDE/non-existent cells) to `PATH` as an STL file. Internal faces between occupied cells are excluded. |
| `--export-merge` | With `--export-stl`, greedy-merge coplanar adjacent faces before triangulating instead of emitting one quad per cell (default). Merging can leave gaps at edges/corners after `--shrink` since neighboring merged faces are shrunk independently; use the per-cell default if you see holes. |
| `--cross-section` | Replace static rendering with an interactive, draggable clip-plane widget so you can see inside the shell/fill. |
| `--no-visualize` | Skip opening the PyVista window (useful for scripted/headless runs). |

Examples:

```powershell
# Shell + solid fill (default)
python src\main.py data\case1.stl --voxel-size 5

# Shell only
python src\main.py data\case1.stl --voxel-size 5 --no-solid-fill

# Solid fill with interactive cross-section
python src\main.py data\case1.stl --voxel-size 5 --cross-section

# Shrink the shell to the STL surface and export the result as STL
python src\main.py data\case1.stl --voxel-size 5 --shrink --export-stl data\output.stl
```

This prints a summary (facet count, bounding box, resulting Voxel count, solid-fill
count, and exported triangle count when `--export-stl` is used) and opens a PyVista
window showing:
- **SHELL** cells (orange, semi-transparent) — voxels touching the STL surface
- **INSIDE** cells (green) — filled interior voxels (unless `--no-solid-fill`)
- Original mesh (light blue wireframe)

## Development commands

```powershell
ruff check src tests          # Lint
ruff format src tests         # Format
mypy src                      # Type check
pytest tests -v               # Run tests
```

## Known limitations

- Without `--cubic-root`, the average-edge-length stopping criterion can give
  anisotropic resolution for very non-cubic bounding boxes.
- Solid fill (default) requires at least one cell of empty space around the shell to
  detect "outside" (handled internally via automatic 1-cell padding) and assumes
  each nested shell layer is a properly closed (watertight) surface; a non-watertight
  mesh can produce incorrect parity results. Use `--no-solid-fill` for a shell-only result.
- `--export-stl` after `--shrink` can produce a non-watertight ("holey") mesh: shrink
  moves each outer face inward independently, so two adjacent outer faces of the same
  cell (e.g. at a corner/edge of the model) can end up offset from each other with no
  geometry filling the gap between them. A true solid union of the Voxel boxes (rather
  than a merged-face STL) is being considered as a follow-up to avoid this.
- `visualize.py` requires a display and is not covered by automated tests.
