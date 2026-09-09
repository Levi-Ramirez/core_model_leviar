# Design: 3D Model Generation

## Pipeline and ownership

```text
pipeline.generate_model(input_path, output_dir, options) -> GenerationResult
    ingest -> validated NIfTI -> segmentation -> validated per-class masks
        -> mesh extraction -> named Scene -> PLY exports
                                         -> rendering -> PNGs
    manifest updated at each stage; owned temporary workspace cleaned
```

`render.py` consumes the assembled Scene, never re-extracts masks, and never
mutates exported geometry. Scene vertices are already in RAS+ millimeters;
node transforms are identity in P0. Export and rendering must apply scene
transforms if nonidentity transforms are supported later.

```text
src/core_model/
    contracts.py     # options, result records, typed errors
    anatomy.py       # supported classes, defaults, context, stable colors
    ingest.py        # discovery, selection, conversion, input validation
    segmentation.py  # inference settings, telemetry disabling, and model invocation
    mesh.py          # mask-to-world extraction
    model.py         # scene assembly and exports
    render.py        # render the assembled meshes without modification
    pipeline.py      # lifecycle, stage orchestration, manifest
    cli.py           # argument parsing and result presentation
```

## API and output contract (R7, R11)

`GenerationOptions` contains `series_id`, `include`, `all_structures`,
`device` (`cpu`, `gpu`, or `mps`), and `quality` (`standard` or `fast`).
Defaults are CPU and standard quality for reproducibility; no hardware
fallback. `include` and `all_structures` are mutually exclusive; an explicit
empty include list is invalid. The task is fixed to `total` in P0.

`list_series(input_path)` returns candidate records with stable selection
IDs, modality, slice count, support status, and rejection reasons.
Geometry support remains unverified until the selected series is converted.
IDs are derived from series UIDs using a collision-checked digest; do not
expose patient tags, free-text descriptions, original filenames, or raw UIDs
in routine output. Identifiers and manifests still remain local artifacts.

`generate_model(...)` returns a `GenerationResult` with status (`complete`
or `partial`), artifact paths, included classes, skipped-class reasons,
warnings, and stage timings. Fatal failures raise a typed `GenerationError`
carrying stage, reason, and any persisted result/manifest path. Library code
does not exit the process or require terminal interaction.

```text
output_dir/
    model.ply
    structures/<class>.ply
    previews/<class>.png
    previews/combined.png
    manifest.json
```

Require a new or empty output directory; reject nonempty directories before
expensive work. P0 has no overwrite mode. Write artifacts through temporary
siblings and atomic replacement. Record only completed artifacts; manifests
have schema version and `running`, `complete`, `partial`, or `failed` status.
A leftover `running` manifest after forced termination is not success.

The pipeline owns a temporary directory outside the repository for extracted
DICOM, converted input, and segmentation masks. Ingest receives this
workspace; its returned paths remain valid until pipeline cleanup in
`finally`. Never delete or modify original inputs. After interruption,
cleanup is best effort; document that process termination/power loss can
leave temporary files. Do not persist intermediates by default.

Geometry/inference/export failures are fatal. Successfully written artifacts
may remain, with a failed manifest; do not present an incomplete export set
as success. Individual preview failures retain all valid exports, attempt
remaining previews, and produce `partial` with per-preview errors. CLI exit
codes: 0 complete, 1 fatal, 2 invalid arguments/selection/input, 3 partial.

Manifest fields include settings, resolved device, versions of all direct
dependencies, model task and weight checksum/identifier, image shape/affine/
units, mask geometry, candidate ID, structure outcomes and colors, clipping
warnings, timings, and relative artifact paths. Exclude
original input paths, patient tags, and raw exception text containing them.

## Input validation and selection (R1)

Choose `dicom2nifti` for conversion and `pydicom` for discovery. Keeping
selection separate makes ambiguity actionable through the same input path.

Group by StudyInstanceUID and SeriesInstanceUID using `pydicom`. P0 supports
conventional single-frame CT; reject non-CT data and enhanced multiframe CT
with explicit reasons. Discovery identifies series, not the best acquisition
or reconstruction for segmentation.

Without an explicit selection, proceed only when exactly one CT series is
present and it passes support checks and conversion. If multiple CT series
are present, list them and require `--series <id>`, even if some are already
known to be unsupported. Provide an executable selection example for each
candidate not already rejected. With no CT series, report that no supported
input was found. Explicit selection cannot override an unsupported input.

Convert only the selected series. Delegate DICOM geometry validation,
including orientation consistency, dimensions, slice spacing, and duplicate
positions, to `dicom2nifti`; keep its validation enabled. Surface converter
failures with actionable, non-identifying reasons. Do not implement our own
orientation classifier, angular threshold, or geometry-based series ranking.
Do not automatically retry another series after conversion failure. Users
can select another series from the same input without manually splitting or
externally converting their files.

Zip extraction rejects absolute paths, traversal, and symlink entries;
extract into the owned workspace. Directory and zip discovery must produce
the same IDs for the same series. Unrelated files are ignored; malformed
DICOM candidates are reported without leaking identifying metadata.

NIfTI input must be a finite, scalar 3D CT volume with at least two samples
per axis and usable physical geometry. Reject 4D, nonfinite, or singular
geometry. Require a coded qform or sform; reject conflicting coded forms
when their corner positions differ by more than 0.01 mm. Convert declared
meter/micron units to mm; reject unknown spatial units. CT/HU provenance
cannot be inferred reliably from arbitrary NIfTI values: document it as an
input obligation rather than silently rescaling intensities.

## Segmentation and model weights (R2, R8)

Use TotalSegmentator's Python API with explicit `task=total`, quality,
device, per-class mask output, and auxiliary previews/statistics disabled.
Run the full task in P0; include selection affects assembly, avoiding an
implicit quality change from ROI-dependent inference. Validate includes
against the tested task label map before calling the model.

Invoke TotalSegmentator directly and allow its normal automatic download of
missing weights. Document that first use requires internet access; cached
weights are managed by TotalSegmentator. Surface download failures with a
clear explanation so the user can retry after restoring connectivity.

Disable usage telemetry through TotalSegmentator's supported configuration
before inference, leaving unrelated settings unchanged. Scan data and
patient metadata remain local. Verify invocation and telemetry disabling
against the tested upstream version. P0 does not add a separate weight-setup
command, application-owned configuration, strict offline enforcement, or
backend concurrency machinery.

Default tests mock inference and downloads and remain offline. Opt-in real
inference may download missing weights, but never downloads scan data.

## Anatomy and failure policy (R3)

`DEFAULT_STRUCTURES` is: `liver`, `spleen`, `kidney_left`, `kidney_right`,
`gallbladder`, `stomach`, `pancreas`, `adrenal_gland_left`,
`adrenal_gland_right`, `duodenum`, `small_bowel`, `colon`, `urinary_bladder`.
Verify this set against the pinned `total` task before release.

`SPINE_AND_PELVIS_STRUCTURES` contains `vertebrae_C1` through `vertebrae_C7`,
`vertebrae_T1` through `vertebrae_T12`, `vertebrae_L1` through `vertebrae_L5`,
`sacrum`, `hip_left`, and `hip_right`. Assemble the union with the selected
interest set, deduplicated by class name. Keep each bone individually named;
this is a set union, not a geometric boolean union.

| Condition | Outcome |
|---|---|
| Unknown include name | Reject before inference with valid-name guidance |
| Expected mask file missing or unreadable | Fatal segmentation-output error |
| Valid empty default/all-class mask | Skip and record `empty_mask`; do not infer absence of anatomy |
| Valid empty explicitly requested mask | Fatal requested-structure error naming the class |
| Some context masks empty | Skip and report unavailable context levels |
| All context masks empty | Export interests with a prominent missing-context warning |
| No nonempty interest masks, even if context exists | Fatal: no anatomy of interest to model |
| Entirely empty scene | Fatal before export |

## Physical geometry and mesh extraction (R4)

Each mask must be a finite, binary 3D image with valid physical units and
its own valid affine. Use its affine to map marching-cubes voxel vertices
into RAS+ mm. Never substitute the source scan affine or apply voxel spacing
twice. P0 requires output masks to match the normalized source shape and
voxel-to-world mapping within 0.01 mm at all eight voxel-grid corners.
Reject different grids explicitly; support for resampled output is future
work. Affines must be finite and invertible. A shared RAS+ convention alone
does not establish alignment across scans; registration remains deferred.

Extract at level 0.5 with unit voxel spacing. Pad nonempty masks with one
zero voxel on every side and subtract that padding offset from vertices
before applying the affine. This closes boundary surfaces. If foreground
touches the original boundary, mark the structure `clipped` and explain that
the cap marks scan coverage, not an observed anatomical boundary. Do not
claim such meshes reconstruct anatomy beyond the scan.

Preserve all components; do not silently remove islands, smooth, or simplify
exports. Ensure outward winding for the chosen binary convention, reverse
winding for a negative affine determinant, and recompute world-space normals.
Reject nonfinite vertices and remove/report degenerate faces consistently.
“Full fidelity” means the un-decimated mask-derived surface, not a claim of
segmentation accuracy or subvoxel anatomical truth.

## Export semantics (R5)

Use trimesh to assemble named geometries and write binary PLY. Export both
the combined mesh and individual class files by default. PLY coordinates
remain RAS+ mm; document import scale and axis handling for external tools.
Keep a versioned color mapping in `anatomy.py`, independent of selection or
iteration order, and write it to the manifest and PLY vertex colors.

Combined PLY concatenates geometry and does not preserve a scene's named
objects. Named individual files provide separate organ inspection in P0.
GLB and other formats are deferred until their names/materials/coordinate
and unit conversions have their own round-trip tests; supporting an export
method in a dependency is not sufficient acceptance evidence.

## Preview rendering (R6)

Use matplotlib `mplot3d` with `Agg` for software rendering without a display
or OpenGL context. Its approximate depth sorting is an accepted limitation;
validate representative combined previews visually before release. Do not
promise correct interpenetrating-surface occlusion or visibility of every
structure in the combined image.

Use an orthographic anterior camera looking along -A, with S up and R to
image-left. Fit bounds with margin while preserving equal units per axis;
never independently stretch axes to fill the image. Output 1200 by 1200 PNGs
with a white background, orientation markers, and a structure title or
combined legend. Use stable organ colors and light gray opaque context;
avoid transparency as a remedy for depth-sorting limitations. Warn in the
combined caption that structures may obscure one another. Individual views
include only the named structure and fit its own extent.

P0 renders the assembled meshes without decimation or face-count budgets.
Do not reject a preview based on mesh size alone or alter source geometry.
Actual rendering failures follow the partial-result policy: retain valid
exports and report the affected preview. Compute-aware optimization is a
potential P1 investigation, motivated by observed resource limitations or
timeouts rather than predetermined face counts. No simplification dependency
or face-reduction accounting is needed in P0.

## Dependencies and device handling (R10, R12)

The README owns the supported-environment information, including operating
systems, Python versions, devices, and validation status. Unavailable
requested devices fail preflight. Performance benchmarking and resource
profiling are potential P1 work; P0 has no runtime or memory release gates.

Declare numpy, nibabel, scikit-image, trimesh, pydicom, dicom2nifti,
TotalSegmentator, and matplotlib as direct dependencies.
Add any further direct imports explicitly. Maintain tested versions and
verify supported wheels in clean environments. Do not rely on transitive
installation or historical install logs.

## Verification and decision sources

Tests and release evidence are enumerated in `tasks.md`. Synthetic fixtures
cover physical geometry, selection, structure identity, clipping, failure
recovery, and rendering. Real-data inference is opt-in and stays local.

The selected libraries keep conversion, geometry, and software rendering
separate. These are design choices to validate, not claims that alternatives
cannot work. Upstream references motivating the contracts:

- [TotalSegmentator usage and telemetry](https://github.com/wasserth/TotalSegmentator):
  task/device/quality settings, weights, and telemetry behavior must be checked
  against the pinned release.
- [NiBabel coordinate systems](https://nipy.org/nibabel/coordinate_systems.html):
  an image's affine maps that image's voxel coordinates to world coordinates.
- [Trimesh scene export implementation](https://github.com/mikedh/trimesh/blob/main/trimesh/exchange/export.py):
  combined PLY export converts a scene into a mesh; named objects are not the
  P0 combined-file contract.
