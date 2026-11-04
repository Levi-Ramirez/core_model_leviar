"""Command-line adapter for model generation and series discovery."""

import argparse
from pathlib import Path
import sys

from core_model.contracts import GenerationError, GenerationOptions
from core_model.generation import generate_model
from core_model.ingest import list_series


def _parser() -> argparse.ArgumentParser:
    """Build the command parser.

    Input: None.
    Output: An argparse parser for the public CLI commands.
    """
    parser = argparse.ArgumentParser(prog="core-model")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate a 3D model from one scan")
    generate.add_argument("input", type=Path)
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--series")
    selection = generate.add_mutually_exclusive_group()
    selection.add_argument("--include", nargs="+", metavar="NAME")
    selection.add_argument("--all-structures", action="store_true")
    generate.add_argument("--device", choices=("cpu", "gpu", "mps"), default="cpu")
    generate.add_argument("--quality", choices=("standard", "fast"), default="standard")
    series = commands.add_parser("list-series", help="list selectable DICOM series")
    series.add_argument("input", type=Path)
    return parser


def _print_candidates(error: GenerationError) -> None:
    """Print safe candidate selection guidance for an ingestion error.

    Input: A generation error, possibly carrying candidate records.
    Output: None; writes safe text to standard error.
    """
    for candidate in getattr(error, "candidates", ()):
        print(f"{candidate.series_id} modality={candidate.modality} slices={candidate.slice_count} "
              f"status={candidate.support_status}", file=sys.stderr)
    for command in getattr(error, "selection_commands", ()):
        print(f"Retry with: {command}", file=sys.stderr)


def main(argv=None) -> int:
    """Run the CLI and return its documented process exit code.

    Input: Optional argument sequence; defaults to ``sys.argv``.
    Output: Exit code: 0 complete, 1 fatal, 2 invalid input/selection,
        or 3 partial preview result.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "list-series":
        try:
            for candidate in list_series(args.input):
                reasons = f" reasons={','.join(candidate.rejection_reasons)}" if candidate.rejection_reasons else ""
                print(f"{candidate.series_id}\t{candidate.modality}\t{candidate.slice_count}\t"
                      f"{candidate.support_status}{reasons}")
            return 0
        except GenerationError as error:
            print(str(error), file=sys.stderr)
            _print_candidates(error)
            return 2
    try:
        options = GenerationOptions(
            series_id=args.series,
            include=tuple(args.include) if args.include is not None else None,
            all_structures=args.all_structures,
            device=args.device,
            quality=args.quality,
        )
        def progress(stage, state):
            print(f"[{stage}] {state}", file=sys.stderr, flush=True)

        result = generate_model(args.input, args.output, options, progress=progress)
    except ValueError as error:
        print(f"invalid arguments: {error}", file=sys.stderr)
        return 2
    except GenerationError as error:
        print(str(error), file=sys.stderr)
        _print_candidates(error)
        return 2 if error.stage in {"generation", "ingest"} else 1
    print(f"status: {result.status}")
    print(f"output: {result.output_dir}")
    print(f"manifest: {result.manifest_path}")
    if result.status == "partial":
        for error in result.errors:
            print(f"warning: {error.stage}:{error.reason}", file=sys.stderr)
        if any(error.reason == "renderer_unavailable" for error in result.errors):
            print("Previews need OpenGL and a display; meshes were still exported. "
                  "See the README for running without a display.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
