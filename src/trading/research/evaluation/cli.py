"""CLI entrypoint for offline model filter evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from trading.research.evaluation.offline_evaluator import run_offline_evaluation
from trading.settings import load_settings
from trading.research.evaluation.purged_cv import PurgedCVConfig
from trading.research.evaluation.report import write_eval_reports
from trading.util.logging import configure_logging, get_logger


def _parse_threshold_grid(s: str) -> tuple[float, ...]:
    """Parse comma-separated floats, e.g. '0.3,0.4,0.5,0.6,0.7'."""
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def main() -> None:
    """Run offline evaluation from CLI."""
    parser = argparse.ArgumentParser(
        description="Offline purged CV evaluation for model filter promotion-readiness.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset path. Supported: .json (decision export), .csv (prepared), .parquet. Default: latest decisions_*.json or prepared_*.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model artifact (.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: TRADING_ARCHIVE_DIR/eval or data/archive/eval)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.3,0.4,0.5,0.6,0.7",
        help="Comma-separated threshold grid (default: 0.3,0.4,0.5,0.6,0.7)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of purged CV folds (default: 5)",
    )
    parser.add_argument(
        "--embargo",
        type=int,
        default=300,
        help="Embargo seconds between train and val (default: 300)",
    )
    parser.add_argument(
        "--purge",
        type=int,
        default=300,
        help="Purge seconds before val window (default: 300)",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=10,
        help="Minimum train samples per fold (default: 10)",
    )
    parser.add_argument(
        "--min-val",
        type=int,
        default=5,
        help="Minimum validation samples per fold (default: 5)",
    )
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="Use rolling window instead of expanding (default: expanding)",
    )
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(
        level=settings.logging.level,
        json_output=settings.logging.json_output,
        logger_name=settings.logging.logger_name,
    )
    logger = get_logger("trading.research.evaluation.cli")

    archive_dir = Path(os.getenv("TRADING_ARCHIVE_DIR", "data/archive"))
    dataset_path = args.dataset
    if dataset_path is None:
        candidates: list[Path] = []
        export_dir = archive_dir / "decision_exports"
        if export_dir.exists():
            candidates.extend(export_dir.glob("decisions_*.json"))
        train_dir = archive_dir / "offline_train"
        if train_dir.exists():
            candidates.extend(train_dir.glob("prepared_*.csv"))
        if candidates:
            dataset_path = max(candidates, key=lambda p: p.stat().st_mtime)
            logger.info("using_latest_dataset", path=str(dataset_path), format=dataset_path.suffix)
    if dataset_path is None:
        logger.error(
            "no_dataset",
            hint="Provide --dataset path. Supported: .json (decision_exports/), .csv (offline_train/prepared_*.csv), .parquet. Run backtest or offline-train first.",
        )
        raise SystemExit(1)

    output_dir = args.output_dir or archive_dir / "eval"
    threshold_grid = _parse_threshold_grid(args.thresholds)
    cv_config = PurgedCVConfig(
        n_splits=args.n_splits,
        embargo_seconds=args.embargo,
        purge_seconds=args.purge,
        min_train_size=args.min_train,
        min_val_size=args.min_val,
        expanding=not args.rolling,
    )

    result = run_offline_evaluation(
        dataset_path=dataset_path,
        model_path=args.model,
        output_dir=output_dir,
        threshold_grid=threshold_grid,
        cv_config=cv_config,
    )

    if not result.success:
        logger.error("eval_failed", error=result.error)
        raise SystemExit(1)

    try:
        written = write_eval_reports(result, output_dir)
        logger.info(
            "eval_reports_written",
            output_dir=str(output_dir),
            files=list(written.keys()),
            total_rows=result.total_rows,
            n_folds=len(result.fold_results),
        )
    except OSError as exc:
        logger.warning("eval_report_write_failed", path=str(output_dir), error=str(exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
