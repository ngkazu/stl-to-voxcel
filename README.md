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

**Scope note**: this produces a *shell* voxelization (only voxels touching the STL
surface), not a solid/filled volume. The STL format itself carries no unit
information — `--voxel-size` is interpreted in whatever unit the STL coordinates use.

## Project layout

```
stl-to-voxcel/
  src/
    main.py              # CLI entry point
    core/
      stl_loader.py       # STL loading (numpy-stl) + degenerate-facet filtering
      geometry.py          # Triangle-box overlap test (SAT, scalar + vectorized batch)
      octree.py             # Octree subdivision -> list[Voxel]
      visualize.py           # PyVista rendering
  tests/                  # pytest unit tests for geometry / stl_loader / octree
  data/                   # Sample/test STL files
  docs/, memos/, references/  # Design notes, planning memos, reference material
  requirements.txt        # numpy, numpy-stl, pyvista
  requirements-dev.txt     # pytest, ruff, mypy
```

## Setup

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

## Usage

```powershell
python src\main.py <path-to-file.stl> --voxel-size <size> [--no-visualize]
```

Example:

```powershell
python src\main.py data\A10.stl --voxel-size 10
```

This prints a summary (facet count, bounding box, resulting Voxel count) and opens a
PyVista window showing the Voxels (orange, semi-transparent) overlaid on the original
mesh (light blue). Pass `--no-visualize` to skip the PyVista window (useful for
scripted/headless runs).

## Development commands

```powershell
ruff check src tests          # Lint
ruff format src tests         # Format
mypy src                      # Type check
pytest tests -v               # Run tests
```

## Known limitations

- Shell-only voxelization; solid (interior-filled) voxelization is a possible future
  extension (see `memos/` for design notes).
- The average-edge-length stopping criterion can give anisotropic resolution for
  very non-cubic bounding boxes.
- `visualize.py` requires a display and is not covered by automated tests.
