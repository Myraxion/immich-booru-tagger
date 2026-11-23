"""Command-line interface entrypoint for immich-booru-tagger."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import datetime

import httpx

from immich_booru_tagger.config import ConfigError, load_config
from immich_booru_tagger.immich import ImmichClient, ImmichError
from immich_booru_tagger.pipeline import (
    DefaultVisionPipeline,
    PipelineError,
    VisionPipeline,
)
from immich_booru_tagger.runner import BatchRunner, RunSummary
from immich_booru_tagger.timefilter import parse_time_bound, validate_time_range


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="immich-booru-tagger",
        description="Automated stateless Booru tagger for Immich albums.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.toml",
        help="Path to the TOML configuration file (default: config.toml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview computed tags without sending any write requests to Immich",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of eligible assets to process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-tag assets even if they already carry meta/booru_tagged",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only process assets createdAt >= TIME (e.g. 7d, 2mo, 1y, 2026-01-01)",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="Only process assets createdAt <= TIME (e.g. 1d, 2026-12-31)",
    )
    return parser


def _print_summary(summary: RunSummary) -> None:
    prefix = "[dry-run] " if summary.dry_run else ""
    for result in summary.results:
        tags_text = ", ".join(result.tags) if result.tags else "(none)"
        print(
            f"{prefix}{result.file_name} ({result.asset_id}) "
            f"[{result.album_name} | {result.mode}]: {tags_text}"
        )

    print(
        f"Summary: albums={summary.albums_resolved}, "
        f"scanned={summary.total_assets_scanned}, "
        f"eligible={summary.total_eligible}, "
        f"processed={summary.processed_count}, "
        f"skipped={summary.skipped_count}, "
        f"batches={summary.batches_count}, "
        f"dry_run={summary.dry_run}"
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    vision_pipeline: VisionPipeline | None = None,
) -> int:
    """CLI main entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    try:
        now = datetime.now().astimezone()
        created_after = (
            parse_time_bound(args.since, now=now, end_of_day=False)
            if args.since is not None
            else None
        )
        created_before = (
            parse_time_bound(args.until, now=now, end_of_day=True)
            if args.until is not None
            else None
        )
        validate_time_range(created_after, created_before)

        config = load_config(args.config)
        active_pipeline: VisionPipeline = (
            vision_pipeline
            if vision_pipeline is not None
            else DefaultVisionPipeline(config.pipeline)
        )
        with ImmichClient.from_config(config, transport=transport) as client:
            runner = BatchRunner(
                config=config,
                client=client,
                vision_pipeline=active_pipeline,
            )
            summary = runner.run(
                dry_run=args.dry_run,
                limit=args.limit,
                force=args.force,
                created_after=created_after,
                created_before=created_before,
            )
    except (ConfigError, ImmichError, PipelineError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
