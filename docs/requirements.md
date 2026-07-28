# Requirements

## Purpose

Given a history of a person's CT scans, build a rotation-corrected 3D model
of their abdominal/pelvic anatomy that's comparable across time — so
scan-to-scan differences reflect real anatomical change, not just how the
patient happened to be lying on the table.

**Self-contained**: a user brings their own raw CT scans and this tool
handles the full pipeline itself, including running
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) — no
separate manual pre-processing step required.

## Scope phases

- **P0 (current)**: a local tool. Runs entirely on the user's own machine —
  "bring your scans" means pointing the CLI/library at local files, nothing
  is transmitted anywhere. This document describes P0 unless noted.
- **Post-P0 (future, not yet started)**:
  - A hosted web interface with upload and view capabilities. Out of scope
    for now — shouldn't shape P0's design beyond keeping the core library UI-agnostic so a
    future web layer can call into it (the way the CLI does).
  - An **"overlap" view**: merge two scans' aligned meshes into one combined
    3D scene so differences can be seen directly, instead of only via
    separate per-scan models and statistics. Deferred until further testing of alignment accuracy.

## Functional requirements (P0)

1. **Input**: raw scan data — a DICOM directory/zip, or a pre-converted
   NIfTI file. The tool performs DICOM→NIfTI
   conversion itself when needed.
2. **Segmentation**: run TotalSegmentator (imported and invoked directly,
   via its Python API — not assumed to have been run already) on the
   NIfTI volume to produce a folder of per-structure masks.
3. **Landmark extraction**: compute world-space (RAS, mm) centroids for a
   predefined set of skeletal structures — the spine (vertebrae) and pelvis
   (sacrum, hips) — from a scan's segmentation output. This set is fixed by
   design, not user-configurable: bone landmarks were empirically validated
   as reliable rigid-registration anchors, while soft-tissue landmarks
   (e.g. the spleen) were not. Every structure in this set must be one
   TotalSegmentator actually outputs as a labeled class — we are not doing
   any segmentation of our own, only consuming TotalSegmentator's, so the
   landmark set is bounded by what its output provides. If a scan's field
   of view doesn't cover some of these structures, extraction uses
   whichever subset is actually present rather than failing outright — but
   the set of *eligible* landmarks itself doesn't change.
4. **Rigid registration**: given matched landmarks from two scans, solve for
   the rotation and translation that best aligns one onto the other. Report
   a measure of fit quality and the magnitude of rotation applied.
5. **Reflection safety**: if the best-fit alignment implies a mirror rather
   than a rotation, correct it to the nearest proper rotation, but surface
   that this happened rather than silently absorbing it.
6. **Volumes**: compute and present to user the physical volume (mL) of any given structure's
   mask.
7. **Laterality**: compute left/right distribution statistics (fraction
   left/right, mean/median position) for a structure after
   applying a registration transform.
8. **Comparison statistics**: given two scans, report relative volume and
   relative position for each organ — with each organ presented separately
   and clearly, not blended into one aggregate number. Position-based
   comparisons must come with an interpretable signal of how much to trust
   them, derived from the registration fit rather than exposed as raw
   technical metrics that aren't meaningful to someone without a background
   in the underlying math.
9. **3D mesh construction**: each scan gets its own independent 3D model —
   its structures of interest plus that same scan's own spine, for
   anatomical context — built entirely from that scan's own data, in its
   own native coordinate frame. P0 does not merge multiple scans into one
   combined/overlaid 3D scene (see non-goals): registration quality isn't
   guaranteed perfect, and a merged view would visually imply more
   precision than the alignment may actually have. Comparison (statistic
   #8) still uses the registration transform internally; the *visual*
   model per scan does not.
10. **Multi-scan history**: not hardcoded to two scans or to any specific
    person's data — the same operations work for N scans, registering each
    to a chosen reference.
11. **CLI**: a command-line entry point covering the full path from raw
    scan input to comparison output, for someone who isn't going to import
    the library directly. This is the primary user-facing surface for P0.

## Non-functional requirements

- **No patient data or PHI in the repo.** `.gitignore` blocks scan
  data (`.nii.gz`, `.dcm`, meshes, output dirs) at the filesystem level, but
  this is also a hard rule for anything written into source, tests, or
  commit messages — tests must use synthetic data only.
- **TotalSegmentator is a heavy dependency, and that's an explicit,
  accepted tradeoff.** It pulls in torch, nnunetv2, and related packages,
  and needs real compute (GPU strongly recommended) plus a one-time model
  weight download. This must be documented clearly in the README install
  instructions.
- **Open-source-ready.** MIT licensed (changeable later — see `LICENSE`).
  Code and comments should be understandable by someone who doesn't have
  this project's chat history — domain terminology (e.g. "registration")
  preferred over invented names.
- **Physically honest math.** Only rigid transforms (rotation + translation)
  are used to compare scans — never deformable/elastic registration, which
  could distort the very differences we're trying to measure. See
  `docs/design.md` for why.
- **UI-agnostic core.** The library (`src/core_model/`) does not assume a
  CLI is the only caller — keep it importable/callable in a way a future
  web backend (post-P0) could reuse without a rewrite. 

## Development practices

These are how we build this, not properties of the delivered system — they
don't affect what a user experiences, only how the codebase stays
trustworthy and maintainable as it grows.

- **Unit tested.** Every core module gets tests with synthetic/generated
  inputs. Tests must run without any external data files or network access
  (aside from the opt-in integration tier below).
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

- Not a diagnostic tool. It does not detect disease, adhesions, or make any
  clinical claims — see the medical disclaimer in the top-level README once
  written.
- No hosted/web component (that's post-P0, unscheduled).
- No GUI in P0 — mesh export (PLY) is the deliverable; viewing happens in
  existing external tools (Blender, MeshLab, etc.) until then.
