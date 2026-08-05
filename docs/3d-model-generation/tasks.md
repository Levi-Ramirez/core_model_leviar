# Tasks: 3D Model Generation

Ordered so each task is buildable and testable on its own, and lands as one
commit (per the Incremental commits practice in `requirements.md`). Later
tasks depend on earlier ones being done; within that constraint, order
follows the pipeline in `design.md`. Each task is split into what gets
built and how it gets verified, so neither side is an afterthought.

- [ ] **Test infrastructure: register the `integration` marker.**
  - Implementation: add the `integration` pytest marker to
    `pyproject.toml`; configure the default `pytest` run to skip tests
    carrying it.
  - Testing: add one throwaway test marked `integration` that would fail if
    run; confirm a plain `pytest` run skips it, and `pytest -m integration`
    picks it up. Remove the throwaway test once confirmed (or leave a real
    one from a later task in its place).

- [ ] **`anatomy.py`**: spine structure-name constant.
  - Implementation: the union of vertebrae, sacrum, and hip
    TotalSegmentator class names (see design's "Structures of interest vs.
    spine"), as a single named constant. No dependency on other modules.
  - Testing: constant is non-empty; contains no duplicates; every entry
    matches TotalSegmentator's class-name naming convention (a static
    format check, not a live lookup against TotalSegmentator itself).

- [ ] **`mesh.py`**: marching-cubes extraction + single-mesh export.
  - Implementation: given a binary mask and its NIfTI affine, run marching
    cubes and transform the resulting vertices into world coordinates;
    export a single mesh via trimesh with the output format as a
    parameter (default PLY). Add `trimesh` to `pyproject.toml`.
  - Testing: synthetic binary masks (e.g. a numpy sphere/cube) with a
    known, computable surface, checked against the extracted mesh; export
    round-trips (re-load and check vertex/face counts, bounding box) for
    at least PLY and one other format.

- [ ] **`ingest.py`**: raw scan input -> NIfTI volume path.
  - Implementation: accept a DICOM directory/zip or an already-NIfTI file
    and normalize either into a single NIfTI volume path. For DICOM input:
    group files by series, classify each series' orientation via
    `pydicom`'s `ImageOrientationPatient` (see design's "Series
    selection"), and keep only axial series; convert with `dicom2nifti`.
    Auto-select and report the choice if exactly one axial series remains;
    raise a clear, candidate-listing error if zero or more than one do.
    Zip contents extracted to a temporary location, never the project
    working directory. Add `dicom2nifti` to `pyproject.toml`.
  - Testing: a tiny synthetic DICOM directory (or a mocked conversion
    call) covering (a) a single axial series — auto-selected and reported,
    (b) an axial series alongside a synthetic sagittal/coronal one —
    reformat correctly discarded, (c) two synthetic axial series — raises
    with both candidates named, and (d) a direct-NIfTI passthrough case;
    no real scan data.

- [ ] **`segmentation.py`**: invoke TotalSegmentator.
  - Implementation: call TotalSegmentator's Python API on a NIfTI volume
    path; return the path to its output segmentation folder. Add
    TotalSegmentator to `pyproject.toml`.
  - Testing: mocked/dependency-injected TotalSegmentator call — assert
    correct invocation arguments and correct handling of its output path,
    without running the real model.

- [ ] **`model.py`**: assemble one scan's exported model.
  - Implementation: given a segmentation folder, resolve
    structures-of-interest (default all, optional include list), add the
    `anatomy.py` spine set, extract each via `mesh.py`, and combine them
    with a `trimesh.Scene` into one exported model.
  - Testing: synthetic per-structure mask fixtures (not real anatomy)
    standing in for a segmentation folder; assert the right structures end
    up in the assembled model for (a) default/all, (b) a narrowed include
    list, and (c) that the spine is present in both cases.

- [ ] **`cli.py`**: command-line entry point.
  - Implementation: wire `ingest.py` -> `segmentation.py` -> `model.py`
    for one scan, with an optional include-list flag and output path.
  - Testing: invoke the CLI with `ingest`/`segmentation` mocked out;
    assert argument parsing and wiring are correct — not re-testing the
    modules' own logic.

- [ ] **Opt-in end-to-end integration test.**
  - Implementation: a test exercising the real pipeline (`ingest.py` ->
    real TotalSegmentator -> `model.py` -> exported file) against a scan
    supplied locally by whoever runs it — never a scan committed to the
    repo (no patient data or PHI rule) and never one fetched automatically
    over the network during the test run. Marked `integration`.
  - Testing: confirm it's excluded from the default `pytest` run and only
    executes under `pytest -m integration`; run it once manually against a
    local scan to confirm the pipeline actually works end to end.

- [ ] **README**: install + usage documentation.
  - Implementation: install instructions covering the TotalSegmentator
    dependency weight/GPU recommendation, and a usage example for the CLI
    entry point.
  - Testing: follow the written instructions on a clean environment (or
    have someone else do so) to confirm they actually work as written, not
    just that they read plausibly.
