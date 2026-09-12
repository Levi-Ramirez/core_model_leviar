# Tasks: 3D Model Generation

Implement the contracts in `requirements.md` and `design.md`. Checkboxes track
completed work; document changes are not evidence of a working pipeline. Land each
module with its tests in reviewable increments. Dependencies follow the
order below; no actual scan, output, or identifying log belongs in Git.

- [x] **Test infrastructure and dependency baseline (R8, R10, R12).**
  - Register `integration`; default pytest excludes it, explicit
    `pytest -m integration` selects it. Confirm both collection behaviors.
  - Establish a tested dependency version set, with direct imports declared
    explicitly. Default tests use synthetic data and deny outbound network;
    heavy inference is mocked, not executed. Generated temporary files are
    allowed. Integration requires an explicitly supplied local scan; a missing scan
    skips with actionable guidance. Real inference may download missing
    weights through TotalSegmentator.
  - Implemented with registered marker/default selection, `pytest-socket`,
    and `local_scan_path` / `--scan-path`. Synthetic subprocess tests verify
    selection, socket controls, and scan prerequisites. The tested current
    dependency set is in `requirements/dev-baseline.txt`; dependencies for
    future modules are added and validated when those modules land.

- [x] **`contracts.py`: options, results, errors (R7, R11).**
  - Implement options, candidate records, structure outcomes, generation
    results, stage errors, and versioned manifest serialization as designed.
  - Test option conflicts, invalid empty includes, result serialization,
    relative artifact paths, and exclusion of identifying input metadata.
  - Implemented using standard-library dataclasses and schema-v1 JSON.
    Tests cover selection snapshots, candidate support status, structure
    outcomes, portable relative paths, failure status consistency, and
    exclusion of raw metadata/exception records from manifests.

- [ ] **`anatomy.py`: defaults, context, label map, colors (R2, R3, R5).**
  - Add the explicit default set, spine-and-pelvis set, stable colors, and a
    versioned supported `total` class map tied to the dependency baseline.
  - Verify actual class membership against the pinned package's local label
    map without loading weights or network. A naming regex is insufficient.
    Test deduplication and color stability across different selections.

- [ ] **`ingest.py`: discover, select, convert, validate (R1, R4, R9).**
  - Implement directory/zip discovery, stable private selection IDs, explicit
    selection, automatic processing only for a single supported CT series,
    and candidate errors with rerun commands. Multiple CT series require
    selection. Implement supported-input checks and safe extraction.
  - Convert only the selected series into the caller-owned workspace. Keep
    converter geometry validation enabled and surface failures without
    retrying other series. Validate the resulting NIfTI dimensions, values,
    forms, units, and affine; do not build a DICOM orientation classifier.
  - Generate a small DICOM stack with known orientation, positions, spacing,
    and rescale values; run the real converter and verify physical landmark
    coordinates and resulting intensities. Mocking alone is insufficient.
  - Test a single supported CT series, multiple series requiring selection
    regardless of orientation, explicit selection, zero CT series, and a
    mixture of supported and unsupported CT series. Verify converter
    rejection propagation and no fallback to another series. Cover non-CT/
    multiframe rejection, malformed inputs, zip traversal/symlinks, and
    equivalent directory/zip IDs. Test that converter validation stays enabled.
  - Test NIfTI passthrough, invalid dimensions, nonfinite values, singular or
    conflicting forms, unknown units, and meter/micron normalization.

- [ ] **`segmentation.py`: inference and model weights (R2, R8, R10).**
  - Invoke TotalSegmentator with explicit task/quality/device/output settings,
    disable usage telemetry using its supported configuration, and validate
    per-class outputs. Leave unrelated settings unchanged and allow normal
    automatic weight downloads; no separate setup command or weight preflight.
  - Verify invocation against the tested upstream API. Test telemetry is
    disabled before inference, download-failure reporting, unavailable-device
    errors, invalid includes, missing outputs, and inference failures. Mock
    inference and downloads so unit tests remain offline.

- [ ] **`mesh.py`: physical surfaces (R4).**
  - Extract at level 0.5 using each mask's affine; implement grid validation,
    unit handling, boundary padding/offset correction, clipped warnings,
    outward winding, and normals. Preserve components and export detail.
  - Test anisotropic spacing, translation, rotation, shear, reflection,
    negative determinant winding, and mask/source grid mismatch using known
    synthetic landmarks. Test empty/all-foreground masks, boundary-touching
    masks, disconnected components, nonbinary values, degenerate geometry,
    and nonfinite affines. Check bounds and outward normals, not only counts.

- [ ] **`model.py`: structure outcomes and PLY artifacts (R3, R5).**
  - Assemble named Scene geometry once, with the designed selection/default/
    all/context semantics. Export combined and named individual binary PLYs
    in common RAS+ mm coordinates with stable vertex colors.
  - Exercise every row of the design's failure-policy table: distinguish
    missing files, empty defaults, empty explicit requests, missing context,
    context-only results, and empty scenes.
  - Round-trip an assembled synthetic model with asymmetric, differently
    colored structures through combined and individual exports. Check
    placement, scale, colors, bounds, and name-to-file manifest mapping.
    Assert scene names in memory; do not assert named objects in combined PLY.

- [ ] **`render.py`: legible previews (R6, R10).**
  - Render the existing Scene into labeled 1200 x 1200 anterior PNGs with
    orthographic projection, equal physical scaling, stable colors, subdued
    context, orientation markers, and the combined occlusion caption.
  - Render assembled meshes without face reduction or face-count budgets.
    Report actual rendering failures and preserve exports; do not omit
    structures or mutate original meshes.
  - Test projected R/S direction, camera depth along A, physical aspect ratio,
    framing of small and elongated structures, labels, and nonblank images.
    A 2D point position alone cannot prove camera depth direction.
  - Test that rendering preserves source vertices, faces, and colors, and
    does not reject scenes based on face count. Visually inspect
    a combined release preview for acceptable occlusion; record limitations
    without patient data.

- [ ] **`pipeline.py`: shared orchestration and lifecycle (R7–R9, R11).**
  - Implement `generate_model`, workspace ownership, preflight, stage manifest
    updates, atomic artifact writes, timings, result assembly, and cleanup.
  - Test new/empty versus nonempty output directories, success, selection
    failure, inference/export failure, interruption, and individual preview
    failure. Verify retained exports and partial status, cleanup, original
    input preservation, artifact inventory, and metadata redaction.
  - Run a synthetic end-to-end pipeline with only inference replaced by
    generated masks, exercising actual conversion, extraction, export,
    rendering, and manifest handling together.

- [ ] **`cli.py`: thin adapter (R1, R7, R8).**
  - Expose `core-model generate INPUT --output DIR`, `--series ID`, mutually
    exclusive `--include NAME ...` / `--all-structures`, `--device`, and
    `--quality`. Expose `core-model list-series INPUT` separately.
  - Test argument parsing, candidate rerun instructions, option forwarding,
    stage summaries, and exit codes 0/1/2/3. The CLI calls the library pipeline;
    it does not duplicate orchestration. No interactive selection is required.

- [ ] **Release integration (R1–R12).**
  - Run a locally supplied scan with actual inference, allowing normal weight
    downloads if needed. Verify geometry, artifact inventory, separate
    structure import, preview usefulness, telemetry disabling, and cleanup.
  - In Blender or MeshLab, check a synthetic asymmetric model's units,
    orientation, colors, and separate structure selection; document import
    settings and application version.

- [ ] **README and package metadata (R5, R8, R10, R12).**
  - Replace stale multi-scan/alignment claims in README and package description
    with actual P0 behavior. Add the visualization disclaimer, direct install
    requirements, supported environments and their validation status,
    device/quality choices, first-run internet access for automatic weight
    downloads, local scan processing, series selection, CLI/library examples,
    output layout, PLY import units, partial-failure semantics, and limitations.
  - Explain dependency/model terms separately from the project's MIT license.
    Correct `.gitignore` wording and cover generated data formats/output dirs;
    ignore rules are assistance, not an enforcement boundary.
  - Follow install and usage instructions in clean target environments and
    confirm examples match the implemented CLI and library API.
