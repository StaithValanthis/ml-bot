"""Offline training CLI entrypoint."""

from __future__ import annotations

import os
from pathlib import Path

from trading.research.training.report import write_offline_train_report
from trading.research.training.runner import run_offline_training
from trading.util.logging import get_logger


def main() -> None:
    """Run offline training on exported decision records."""
    from trading.util.logging import configure_logging

    from trading.settings import load_settings

    settings = load_settings()
    configure_logging(
        level=settings.logging.level,
        json_output=settings.logging.json_output,
        logger_name=settings.logging.logger_name,
    )
    logger = get_logger("trading.research.training.run_offline")

    archive_dir = Path(os.getenv("TRADING_ARCHIVE_DIR", "data/archive"))
    export_dir = archive_dir / "decision_exports"
    export_dir_resolved = export_dir.resolve()
    records_path: Path | None = None
    if export_dir.exists():
        json_files = sorted(export_dir.glob("decisions_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if json_files:
            records_path = json_files[0]
            logger.info(
                "offline_train_using_export",
                path=str(records_path.resolve()),
                export_dir=str(export_dir_resolved),
                count=len(json_files),
            )
    if records_path is None:
        export_exists = export_dir.exists()
        json_count = len(list(export_dir.glob("decisions_*.json"))) if export_exists else 0
        logger.warning(
            "offline_train_no_export_found",
            export_dir=str(export_dir_resolved),
            export_dir_exists=export_exists,
            decisions_json_count=json_count,
            expected_pattern="decisions_*.json",
            hint="Run backtest first: trading-backtest or TRADING_MODE=backtest python -m trading.main",
        )
        return

    output_dir = archive_dir / "offline_train"
    result = run_offline_training(
        records_path=records_path,
        output_dir=output_dir,
    )
    try:
        json_path, md_path = write_offline_train_report(result, archive_dir)
        logger.info(
            "offline_train_report_written",
            json_path=str(json_path),
            md_path=str(md_path),
            success=result.success,
            train_rows=result.train_rows,
            test_rows=result.test_rows,
        )
    except OSError as exc:
        logger.warning("offline_train_report_failed", path=str(archive_dir), error=str(exc))


if __name__ == "__main__":
    main()
