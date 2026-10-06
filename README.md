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

This produces a *shell* voxelization (only voxels touching the STL surface). Pass
`--solid-fill` to additionally fill the interior: the shell is rasterized onto a
uniform grid, empty cells are grouped into 6-connected regions, and a region-adjacency
parity walk (even/odd number of shell layers crossed from the outside) decides which
regions are solid material (filled) versus nested interior cavities (left empty). See
`src/core/solid_fill.py` for details.

Pass `--expand <distance>` to add an outer margin: the voxel mesh is morphologically
dilated and then validated so that every exterior vertex is at least `<distance>` away
from the original STL surface. This uses a KD-tree accelerated distance calculation
for performance. See `src/core/expand.py` for details.

The STL format itself carries no unit information — `--voxel-size` and `--expand` are
interpreted in whatever unit the STL coordinates use.

## Project layout

```
stl-to-voxcel/
  src/
    main.py              # CLI entry point
    core/
      model.py            # Data structures: Cell, CellState, VoxelModel
      stl_loader.py       # STL loading (numpy-stl) + degenerate-facet filtering
      geometry.py         # Triangle-box overlap test, point-to-mesh distance (KD-tree)
      octree.py           # Octree subdivision -> VoxelModel (SHELL cells)
      solid_fill.py       # Interior fill via flood-fill + shell-crossing parity (INSIDE cells)
      expand.py           # Morphological expansion with distance validation (OUTSIDE cells)
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
| `--solid-fill` | Fill the interior (material) in addition to the shell; nested cavities are correctly left empty (see "What it does" above). |
| `--expand <distance>` | Expand the voxel mesh outward, ensuring all exterior vertices are at least `<distance>` away from the STL surface. Implies `--solid-fill`. |
| `--cross-section` | Replace static rendering with an interactive, draggable clip-plane widget so you can see inside the shell/fill. |
| `--no-visualize` | Skip opening the PyVista window (useful for scripted/headless runs). |

Examples:

```powershell
# Basic shell voxelization
python src\main.py data\case1.stl --voxel-size 5

# Solid fill with interior
python src\main.py data\case1.stl --voxel-size 5 --solid-fill --cross-section

# Expand outward by 3 units
python src\main.py data\case1.stl --voxel-size 5 --expand 3
```

This prints a summary (facet count, bounding box, resulting Voxel count, and solid-fill
count when requested) and opens a PyVista window showing:
- **SHELL** cells (orange, semi-transparent) — voxels touching the STL surface
- **INSIDE** cells (green) — filled interior voxels (when `--solid-fill` or `--expand`)
- **OUTSIDE** cells (blue) — expanded margin voxels (when `--expand`)
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
- `--solid-fill` requires at least one cell of empty space around the shell to
  detect "outside" (handled internally via automatic 1-cell padding) and assumes
  each nested shell layer is a properly closed (watertight) surface; a non-watertight
  mesh can produce incorrect parity results.
- `--expand` validates that exterior *vertices* are at least the specified distance
  from the STL surface. Cell edges/faces may be slightly closer in some cases.
- `visualize.py` requires a display and is not covered by automated tests.
