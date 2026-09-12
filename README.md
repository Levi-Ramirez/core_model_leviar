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

## Supported environments

The planned 3D model generation pipeline is not implemented yet, so these
are validation targets, not verified support claims. Update this table with
tested versions and device configurations as implementation is validated.

| Planned environment | Planned devices | Status |
|---|---|---|
| Ubuntu 22.04, x86-64, Python 3.10 | CPU, NVIDIA GPU with CUDA | Unverified |
| macOS 14+, Apple Silicon, Python 3.10 | CPU, MPS | Unverified |
| Windows 11, x86-64, Python 3.10 | CPU | Unverified |

Performance benchmarking and hardware-specific resource targets are potential
P1 work; no runtime or memory guarantees are currently established.

## Test

Create an isolated development environment and install the tested dependency
baseline for the current modules:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements/dev-baseline.txt -e ".[dev]"
```

The baseline was tested with CPython 3.10.4 on macOS 26.5.2 (Apple Silicon).
It covers registration and test infrastructure; model-generation dependencies
will be added and validated with their modules. It is not a cross-platform
lockfile or evidence that the planned pipeline is supported yet.

Run synthetic unit tests (no real inference):

```bash
pytest
```

Default runs exclude the registered `integration` marker. `pytest-socket`
blocks Python socket access during tests, including DNS. This guards against
accidental requests; it is not an OS-level sandbox for subprocesses or native
libraries. Keep test imports free of network access and model execution.

Future real-inference tests must use `@pytest.mark.integration` and the
`local_scan_path` fixture. Supply a scan explicitly:

```bash
pytest -m integration --scan-path /path/to/local/scan
```

Without `--scan-path`, tests using that fixture skip with guidance. A supplied
path that does not exist is an error. Never add real scans to the repository.
Integration tests that need model-weight downloads also use
`@pytest.mark.enable_socket`; network access remains blocked otherwise.
Mock inference and downloads in unit tests.

No real-inference integration tests exist yet. The test-infrastructure suite
verifies selection, network controls, and scan-input handling with temporary
synthetic test projects. A direct `pytest -m integration` currently selects
no tests (pytest exit code 5).

## License

MIT — see `LICENSE`.
