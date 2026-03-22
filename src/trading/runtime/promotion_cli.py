"""CLI entrypoint for promotion-readiness evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from trading.runtime.promotion_readiness import (
    DEFAULT_MINIMUM_MODEL_EVALUATIONS,
    DEFAULT_MINIMUM_PASSING_SESSIONS,
    DEFAULT_MINIMUM_TOTAL_DURATION_SECONDS,
    DEFAULT_MINIMUM_TOTAL_FILLS,
    run_promotion_evaluation,
)
from trading.util.logging import configure_logging, get_logger


def main() -> None:
    """Run promotion-readiness evaluation from CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate soak reports for promotion readiness (demo -> paper -> next phase).",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        dest="inputs",
        metavar="PATH",
        help="Soak report path or glob (e.g. data/archive/session_summaries/soak_report_*.json). May be repeated.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to scan for soak_report_*.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: TRADING_ARCHIVE_DIR/promotion or data/archive/promotion)",
    )
    parser.add_argument(
        "--min-passing-sessions",
        type=int,
        default=DEFAULT_MINIMUM_PASSING_SESSIONS,
        help=f"Minimum passing sessions for READY_FOR_PAPER (default: {DEFAULT_MINIMUM_PASSING_SESSIONS})",
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=DEFAULT_MINIMUM_TOTAL_DURATION_SECONDS,
        help=f"Minimum total duration in seconds (default: {DEFAULT_MINIMUM_TOTAL_DURATION_SECONDS})",
    )
    parser.add_argument(
        "--min-fills",
        type=int,
        default=DEFAULT_MINIMUM_TOTAL_FILLS,
        help=f"Minimum total fills (default: {DEFAULT_MINIMUM_TOTAL_FILLS})",
    )
    parser.add_argument(
        "--min-model-evaluations",
        type=int,
        default=DEFAULT_MINIMUM_MODEL_EVALUATIONS,
        help=f"Minimum total model evaluations (default: {DEFAULT_MINIMUM_MODEL_EVALUATIONS})",
    )
    args = parser.parse_args()

    if not args.inputs and args.dir is None:
        parser.error("Provide --input and/or --dir")

    from trading.settings import load_settings

    settings = load_settings()
    configure_logging(
        level=settings.logging.level,
        json_output=settings.logging.json_output,
        logger_name=settings.logging.logger_name,
    )
    logger = get_logger("trading.runtime.promotion_cli")

    archive_dir = Path(os.getenv("TRADING_ARCHIVE_DIR", "data/archive"))
    output_dir = args.output_dir or archive_dir / "promotion"

    json_path, md_path, assessment = run_promotion_evaluation(
        input_specs=args.inputs,
        dir_path=args.dir,
        output_dir=output_dir,
        minimum_passing_sessions=args.min_passing_sessions,
        minimum_total_duration_seconds=args.min_duration_seconds,
        minimum_total_fills=args.min_fills,
        minimum_model_evaluations=args.min_model_evaluations,
        logger=logger,
    )

    verdict = assessment.get("promotion_verdict", {}).get("verdict", "NOT_READY")
    logger.info("promotion_eval_complete", verdict=verdict, json_path=str(json_path), markdown_path=str(md_path))
    print(f"Verdict: {verdict}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
