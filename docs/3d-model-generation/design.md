# Design: 3D Model Generation

## Pipeline

```
raw scan (DICOM dir/zip, or NIfTI)
    │  ingest.py
    ▼
NIfTI volume
    │  segmentation.py  (invokes TotalSegmentator)
    ▼
folder of per-structure mask files (.nii.gz)
    │  mesh.py            (marching cubes, per structure)
    ▼
per-structure meshes, in world (RAS, mm) coordinates
    │  model.py            (assemble structures-of-interest + spine)
    ▼
one exported 3D model per scan
```

## Module layout

```
src/core_model/
├── ingest.py         # raw scan (DICOM/NIfTI) -> NIfTI volume path
├── segmentation.py     # invoke TotalSegmentator -> segmentation folder
├── anatomy.py            # structure-name constants (see below)
├── mesh.py                  # marching-cubes extraction + export, per structure
├── model.py                    # assemble a scan's structures-of-interest + spine into one model
└── cli.py                         # command-line entry point
```

## Structures of interest vs. spine

A model is built from two kinds of structures:

- **Structures of interest**: defaults to _all_ structures TotalSegmentator
  produced for the scan. A user can narrow this with an explicit include
  list (e.g. just `liver`, `colon`) — each name must be a TotalSegmentator
  output class.
- **Spine**: fixed context set, always included regardless of the
  structures-of-interest selection — even a narrowed include list still
  gets the spine added, so the model is anatomically orientable on its own.
  Defined as the union of whichever vertebrae, sacrum, and hip masks
  TotalSegmentator actually produced for that scan (not all scans' fields of
  view include every level).

The spine structure-name list lives in `anatomy.py` as a named constant,
rather than inline in `model.py`, so "what counts as the spine" has one
definition to update if TotalSegmentator's label set ever changes.

## Mesh extraction

Marching cubes (`skimage.measure.marching_cubes`) on each structure's
binary mask, at the 0.5 isosurface level — the standard approach for
converting a label volume into a surface mesh, already proven out in this
project's earlier ad hoc analysis scripts. Resulting vertices are in voxel
index space; they're transformed into world coordinates (RAS, mm) via the
scan's NIfTI affine (`nibabel`) before export, so the model reflects
physical geometry rather than raw array indices.

## Coordinate conventions

World coordinates are RAS+ (Right-Anterior-Superior positive), the same
convention `registration.py` uses. This feature doesn't apply any
registration transform to its meshes, but keeping the same coordinate
convention means these meshes are usable as-is by the post-P0 overlap view
once that's built, without a rewrite.

## Mesh export format

The job this decision covers is narrow: `skimage.measure.marching_cubes`
(already chosen — see Mesh extraction above) hands back vertex and face
arrays; something needs to turn those into an exported file. It is not a
choice of marching-cubes implementation, and not a choice about stacking
2D slices into a volume — this project's input is already a 3D NIfTI
volume via `nibabel`, so there's no slice-stacking step to have an opinion
about.

| Option                       | Pros                                                                                                                                                                                                                                                                                                                                                                                                     | Cons                                                                                                                                                                                                                                                                             |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **trimesh** (chosen)         | Numpy-based, pip-installs cleanly with no compilation; takes `marching_cubes` vertex/face arrays directly; binary PLY (meaningfully smaller than ASCII for organ-scale meshes, which can run into hundreds of thousands of triangles) plus OBJ/STL/GLB for free; `trimesh.Scene` is a direct fit for assembling structures-of-interest + spine into one exported model; actively maintained, widely used | A new dependency to track/pin; adds an abstraction layer between us and the raw file bytes; some optional trimesh features pull in extra dependencies (core PLY/OBJ/STL export does not)                                                                                         |
| **Open3D**                   | Comprehensive 3D toolkit (registration, point clouds, built-in viewer)                                                                                                                                                                                                                                                                                                                                   | Heavy dependency — compiled C++ core, large wheels, bundles a visualization stack we don't need; historically platform-dependent install friction; most of its capability goes unused for a narrow export job                                                                    |
| **PyMesh**                   | Purpose-built for mesh processing (boolean ops, repair) that could matter for future features                                                                                                                                                                                                                                                                                                            | Weak pip-installability — often needs building from source against system libraries (e.g. CGAL); less actively maintained; fragile dependency for a project meant to be easy for others to install                                                                               |
| **Hand-rolled PLY exporter** | Zero new dependencies; full control; nothing external to break                                                                                                                                                                                                                                                                                                                                           | We own every bug/edge case; ASCII-only unless we also hand-roll binary PLY (materially more complex — correct little-endian struct packing per the format spec); no OBJ/STL/GLB without writing each ourselves; no built-in multi-mesh scene composition — we'd rebuild that too |

**Chosen: trimesh.** It's the lightest option that still does the actual
job (format export + scene composition) without Open3D's install weight or
PyMesh's install fragility, and it avoids reimplementing binary PLY
correctness by hand.

**Output format: PLY by default**, matching the deliverable stated in
requirements. The exporter is not hardcoded to PLY-only, though — trimesh
makes OBJ/STL/GLB available at effectively no extra cost, and GLB in
particular is the standard format a future web viewer (post-P0) would
want, so the export function takes the format as a parameter rather than
assuming PLY unconditionally. P0's CLI defaults to PLY and does not expose
a format flag as a feature in its own right.

## Testing strategy

- `mesh.py`: unit tests build tiny synthetic binary masks (e.g. a numpy
  sphere/cube) with a known, computable surface — no real segmentation
  output or TotalSegmentator involved.
- `model.py`: unit tests assemble synthetic per-structure meshes (not real
  anatomy) into a model and check the right structures ended up in it.
- `segmentation.py`: tested via mocking/dependency injection — verifies
  correct invocation and output handling without running the real model.
- `ingest.py`: tested against a tiny synthetic DICOM directory or a mocked
  conversion call, not a real scan.
- Real end-to-end runs (actual TotalSegmentator, actual scan) live in the
  opt-in integration test tier, not the default suite.
