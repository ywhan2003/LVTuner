import argparse
import os
from pathlib import Path
from typing import Sequence

from functions import TuningEntrypoint, available_pipelines, load_run_pipeline


REPO_ROOT = Path(__file__).resolve().parent


def _resolve_config_path(
    config_arg: str | None,
    entrypoint: TuningEntrypoint,
    invocation_cwd: Path,
    repo_root: Path = REPO_ROOT,
) -> str:
    if not config_arg:
        return str((repo_root / entrypoint.default_config).resolve())

    raw_path = Path(config_arg).expanduser()
    if raw_path.is_absolute():
        return str(raw_path)

    cwd_path = (invocation_cwd / raw_path).resolve()
    if cwd_path.exists():
        return str(cwd_path)

    return str((repo_root / raw_path).resolve())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RFANNSTuner tuning pipelines.")
    subparsers = parser.add_subparsers(dest="pipeline", required=True)

    for entrypoint in available_pipelines():
        subparser = subparsers.add_parser(entrypoint.name, help=entrypoint.description)
        subparser.set_defaults(entrypoint=entrypoint)
        subparser.add_argument(
            "--config",
            type=str,
            default=None,
            help=f"Path to YAML config. Default: {entrypoint.default_config}",
        )
        subparser.add_argument(
            "--resume",
            dest="resume",
            action="store_true",
            default=True,
            help="Resume from existing trials/<trials_name>.jsonl (default: enabled).",
        )
        subparser.add_argument(
            "--no-resume",
            dest="resume",
            action="store_false",
            help="Do not resume from existing trials.",
        )
        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help="Generate candidates and commands only; do not execute benchmark.",
        )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    invocation_cwd = Path.cwd()
    args = parse_args(argv)
    entrypoint: TuningEntrypoint = args.entrypoint
    config_path = _resolve_config_path(
        config_arg=args.config,
        entrypoint=entrypoint,
        invocation_cwd=invocation_cwd,
    )

    os.chdir(REPO_ROOT)
    try:
        run_pipeline = load_run_pipeline(entrypoint.name)
        return int(run_pipeline(config_path=config_path, resume=args.resume, dry_run=args.dry_run))
    finally:
        os.chdir(invocation_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
