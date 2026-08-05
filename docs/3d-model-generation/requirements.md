# Requirements: 3D Model Generation

## Purpose

Given one of a person's CT scans, build a 3D model of their
abdominal/pelvic anatomy of interest, including their spine for anatomical
context, that can be viewed and inspected outside a medical imaging
workstation.

**Self-contained**: a user brings their own raw CT scan and this feature
handles the full pipeline itself, including running
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) — no
separate manual pre-processing step required.

## Scope phases

- **P0 (current)**: a local tool. Runs entirely on the user's own machine —
  "bring your scans" means pointing the CLI/library at local files, nothing
  is transmitted anywhere.
- **Post-P0 (future, not yet started)**:
  - A hosted web interface with upload and view capabilities.
  - An **"overlap" view**: merge two scans' meshes into one combined 3D
    scene so differences can be seen directly, instead of only via separate
    per-scan models. Deferred until further testing of alignment accuracy.
  - **Multi-scan / batch invocation**: process a whole scan history in one
    call. P0's CLI processes one scan per invocation; producing models for
    several scans means invoking it once per scan.

## Functional requirements

1. **Input**: raw scan data — a DICOM directory/zip, or a pre-converted
   NIfTI file. The tool performs DICOM→NIfTI conversion itself when needed.
2. **Segmentation**: run TotalSegmentator (imported and invoked directly,
   via its Python API — not assumed to have been run already) on the
   NIfTI volume to produce a folder of per-structure masks.
3. **3D mesh construction**: each scan gets its own independent 3D model —
   its structures of interest plus that same scan's own spine, for
   anatomical context — built entirely from that scan's own data, in its
   own native coordinate frame.
4. **CLI**: a command-line entry point covering the full path from raw
   scan input to an exported 3D model, for someone who isn't going to
   import the library directly. Operates on one scan per invocation.

## Non-functional requirements

- **No patient data or PHI in the repo.** `.gitignore` blocks scan data
  (`.nii.gz`, `.dcm`, meshes, output dirs) at the filesystem level, but this
  is also a hard rule for anything written into source, tests, or commit
  messages — tests must use synthetic data only.
- **TotalSegmentator is a heavy dependency, and that's an explicit,
  accepted tradeoff.** It pulls in torch, nnunetv2, and related packages,
  and needs real compute (GPU strongly recommended) plus a one-time model
  weight download. This must be documented clearly in the README install
  instructions.
- **Open-source-ready.** MIT licensed (changeable later — see `LICENSE`).
  Code and comments should be understandable by someone who doesn't have
  this project's chat history — domain terminology preferred over invented
  names.
- **UI-agnostic core.** The library (`src/core_model/`) does not assume a
  CLI is the only caller — keep it importable/callable in a way a future
  web backend (post-P0) could reuse without a rewrite.

## Development practices

These are how we build this, not properties of the delivered system — they
don't affect what a user experiences, only how the codebase stays
trustworthy and maintainable as it grows.

- **Unit tested.** Every module gets tests with synthetic/generated inputs.
  Tests must run without any external data files or network access (aside
  from the opt-in integration tier below).
- **Two-tier testing.** Fast unit tests (the default `pytest` run) use only
  synthetic, in-memory data and must not invoke the real TotalSegmentator
  model — the segmentation-invocation module is tested via mocking/
  dependency injection (correct arguments, correct output handling, correct
  DICOM→NIfTI conversion logic) instead. A separate, explicitly-marked slow/
  integration tier (e.g. `pytest -m integration`) may exercise the real
  model end-to-end, but is opt-in and not part of default CI-style runs.
- **Incremental commits.** Each module lands in its own commit alongside its
  tests.

## Explicit non-goals (for P0)

- Not a diagnostic tool. A 3D model is a visualization, not a clinical
  finding. See the medical disclaimer in the top-level README once written.
- No hosted/web component (that's post-P0, unscheduled).
- No GUI in P0 — exported mesh files (PLY by default) are the deliverable;
  viewing happens in existing external tools (Blender, MeshLab, etc.) until
  then.
- No merged/overlaid multi-scan 3D visualization in P0 (that's the post-P0
  "overlap" view) — each scan's 3D model stands alone.
- No batch/multi-scan processing in one CLI invocation in P0 (that's
  post-P0) — one scan per invocation.
