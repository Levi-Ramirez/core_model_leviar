# Requirements: 3D Model Generation

## Purpose and scope

Build an inspectable 3D model of abdominal/pelvic anatomy from one person's
local CT scan, including that scan's spine and pelvis for context. The tool
handles supported raw DICOM input, conversion, TotalSegmentator inference,
mesh export, and static previews without manual preprocessing.

P0 is one scan per invocation, with a CLI and reusable Python library. Each
model remains in its scan's native physical frame. Hosted uploads, batch
processing, registration/overlap views, and interactive viewers are future
work. A common coordinate convention does not align different scans.

**Potential P1 — multi-series mesh averaging (exploratory, not committed).**
Explore running TotalSegmentator on multiple CT series from the same person
and combining their meshes into an averaged model. This requires evaluating
alignment, differences in coverage and acquisition, and what mesh averaging
should mean before specifying or implementing it. It is outside P0.

**Potential P1 — previews adapted to available compute (exploratory, not
committed).** Consider preview optimization, including mesh simplification,
only in response to demonstrated compute limitations or rendering timeouts
on users' machines. P0 has no face-reduction logic or arbitrary face budgets.

**Potential P1 — performance evaluation (exploratory, not committed).**
Consider representative-scan benchmarks, runtime and peak-memory profiling,
and hardware-specific performance targets. P0 does not require performance
benchmarks or runtime/memory release gates.

This is a visualization tool, not a diagnostic tool or clinical finding.

## Functional requirements

- **R1 — Input and selection.** Accept a DICOM directory/zip or a 3D NIfTI
  (`.nii` or `.nii.gz`). List DICOM series and allow explicit selection
  through both CLI and library. Automatically proceed only when the input
  contains a single supported CT series. When multiple CT series are
  present, list them and require explicit selection; do not rank or discard
  them by orientation. Let the converter validate the selected series'
  supported geometry and report conversion failures with specific reasons.
  Do not silently try a different series after failure. P0 supports
  conventional single-frame CT; reject non-CT DICOM and enhanced multiframe
  CT. NIfTI inputs must be finite scalar 3D CT volumes with valid physical
  geometry and known spatial units.
- **R2 — Segmentation.** Invoke TotalSegmentator's Python API directly with
  explicit task, quality, device, and per-class output settings. P0 uses
  task `total`, standard resolution by default, and optional fast quality.
  Validate class names before inference and validate returned masks.
- **R3 — Anatomy.** Default to the liver, spleen, kidneys, gallbladder,
  stomach, pancreas, adrenal glands, duodenum, small bowel, colon, and
  urinary bladder. Support an explicit include list or an all-classes
  option. Always request spine-and-pelvis context: cervical, thoracic, and
  lumbar vertebrae, sacrum, and both hip bones. Each included structure
  appears exactly once.
  Report empty, skipped, and unavailable structures distinctly; never
  silently replace a failed requested structure with context alone.
- **R4 — Physical geometry.** Extract from each mask using its own affine,
  normalize physical units to millimeters, and preserve native RAS+
  coordinates. Reject invalid or inconsistent geometry. Define clipping
  and empty-mask behavior, preserve disconnected components, and avoid
  smoothing or decimating exported meshes in P0.
- **R5 — Exports and identity.** Produce a combined binary `model.ply` plus
  named `structures/<class>.ply` files, allowing organs to be imported and
  selected separately. All files use the same coordinates, scale, and
  stable structure colors. A manifest records names, units, and paths.
  Combined PLY does not promise named objects or embedded unit metadata.
- **R6 — Previews.** Produce one PNG per included structure, including
  context structures, plus one combined PNG. Use a fixed shaded
  anterior-oblique view, equal physical scaling, content-based framing, stable
  colors, structure labels/legend, and a camera-matched orientation guide.
  Individual views include available translucent spine/pelvis reference in the
  same physical coordinates; selected bones use opaque light blue. Explain
  approximate surface overlap in the caption. Render the assembled meshes
  without face reduction in P0 and leave their geometry unchanged; hidden
  structures remain inspectable through their individual previews and exports.
- **R7 — Invocation and results.** CLI and library expose the same complete
  pipeline. Return structured output paths, included/skipped structures,
  warnings, stage status, and errors. Preserve original inputs, keep owned
  temporary files available for the duration of processing, and clean them
  on completion or failure. Reject nonempty output directories before
  expensive work. Inference, geometry, and export failures are fatal;
  preview failures retain valid exports and report partial success with
  the affected previews identified. Never report incomplete output as a
  complete run.

## Operational requirements

- **R8 — Local processing.** Scan data and patient metadata stay local.
  Disable TotalSegmentator usage telemetry before inference. Allow its
  normal automatic download of missing model weights; document that the
  first run requires internet access and report download failures clearly.
  P0 does not promise completely offline execution or require a separate
  weight-setup command.
- **R9 — Data handling.** No patient data or PHI in tracked files, fixtures,
  source, logs committed to Git, or commit messages. `.gitignore` helps
  exclude untracked artifacts from normal Git operations; it does not
  prevent filesystem writes or guarantee that data cannot be committed.
  Use generated fixtures in the default suite. Real scans are permitted
  only as locally supplied inputs to opt-in integration checks. Avoid
  patient identifiers and free-text DICOM fields in generated filenames,
  logs, and manifests. Temporary files must be cleaned on success/failure.
- **R10 — Supported operation.** State supported operating systems, Python
  versions, and compute devices in the README, distinguishing tested support
  from planned targets. Document dependency weight, device/quality choices,
  and errors for unavailable devices. No silent change of quality or device.
- **R11 — Reproducibility.** Record application/dependency/model versions,
  weight identifiers or checksums, resolved settings/device, source and
  mask geometry, structure outcomes, stage timings,
  and artifact paths in a local manifest without patient-identifying tags.
- **R12 — Packaging.** Keep an importable, UI-agnostic core. Declare every
  directly imported third-party dependency explicitly and maintain a tested
  version set. Project code is MIT licensed; document dependency/model
  terms separately. Documentation must stand without chat history.

## Development and acceptance

Each implementation task lands with its meaningful synthetic tests. Default
`pytest` runs offline without real inference; generated files in temporary
directories are allowed. Real inference checks are marked
`integration`, opt-in via `pytest -m integration`, and use locally provided
scans. They never download scans; real inference may download missing model
weights through TotalSegmentator.

Acceptance covers physical coordinates, assembled export round trips,
selection and failure behavior, telemetry disabling, preview geometry, and
cleanup—not merely non-empty files or mocked successful calls. Each
requirement must have corresponding verification evidence. Incremental
commits should keep each module and its tests reviewable together.
