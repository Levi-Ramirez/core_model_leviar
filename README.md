# core_model_leviar

Build a rotation-corrected 3D model of your abdominal/pelvic anatomy across a
history of CT scans.

Takes [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) output
(one folder of organ masks per scan) and:

1. Rigidly registers scans to each other using bone landmarks (Kabsch
   algorithm — rotation + translation, no deformation), so scans taken with
   the patient in slightly different positions can be compared honestly.
2. Reports registration quality (RMSE, rotation angle) so you know how much
   to trust the alignment.
3. Computes organ volumes and left-right ("laterality") distributions.
4. Exports aligned 3D meshes (PLY) of chosen structures across time, so you
   can view your own anatomical history as a single 3D scene.

Status: early / personal project. See `docs/design-notes.md` (WIP) for the
reasoning behind the landmark and registration choices.

## Install

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest
```

## License

MIT — see `LICENSE`.
