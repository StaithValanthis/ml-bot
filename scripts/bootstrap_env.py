#!/usr/bin/env python3
"""
Interactive .env bootstrap for ML Bot.

Prompts for configuration, validates inputs, and writes a clean .env file.
Secrets are masked; never echoed.
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
from pathlib import Path

# Valid runtime modes (backtest-ready => paper mode for runtime, testnet env)
MODES = ("paper", "demo", "live", "backtest-ready")
MODE_TO_ENV: dict[str, str] = {
    "paper": "bybit_testnet",  # paper can use testnet for market data
    "demo": "bybit_demo",      # DEMO preferred over testnet
    "live": "bybit_mainnet",
    "backtest-ready": "bybit_testnet",
}
# Actual TRADING_MODE written to .env (backtest-ready runs as paper)
MODE_TO_TRADING_MODE: dict[str, str] = {
    "paper": "paper",
    "demo": "demo",
    "live": "live",
    "backtest-ready": "paper",
}
LIVE_CONFIRM_PHRASE = "ENABLE LIVE TRADING"
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]+USDT$")


def _prompt(prompt: str, default: str | None = None, secret: bool = False) -> str:
    """Prompt user; use getpass for secrets."""
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    if secret:
        return getpass.getpass(prompt).strip() or (default or "")
    return input(prompt).strip() or (default or "")


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt for yes/no; return bool."""
    default_str = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Enter y/yes or n/no")


def _validate_symbol(s: str) -> bool:
    return bool(SYMBOL_PATTERN.match(s.strip()))


def _parse_symbols(raw: str) -> list[str]:
    """Parse comma-separated symbols; validate format."""
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    invalid = [s for s in symbols if not _validate_symbol(s)]
    if invalid:
        raise ValueError(f"Invalid symbols (must be *USDT): {invalid}")
    return symbols


def _validate_mode(mode: str) -> str:
    m = mode.strip().lower()
    if m not in MODES:
        raise ValueError(f"Mode must be one of: {', '.join(MODES)}")
    return m


def run_interactive(
    *,
    output_path: Path,
    install_path: Path,
    default_mode: str = "paper",
    default_dry_run: bool = True,
) -> None:
    """Run interactive prompts and write .env."""
    lines: list[str] = []

    # Mode
    print("\n--- Runtime mode ---")
    print("  paper         = live market data, no exchange orders")
    print("  demo          = Bybit demo (api-demo.bybit.com), exchange-connected")
    print("  live          = real money, mainnet (requires confirmation)")
    print("  backtest-ready = config for backtest; runtime uses paper")
    mode = _prompt("Mode", default_mode)
    try:
        mode = _validate_mode(mode)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Live confirmation
    if mode == "live":
        print("\n*** WARNING: LIVE MODE = REAL MONEY ***")
        print("You must type the exact phrase to continue.")
        confirm = _prompt(f"Type '{LIVE_CONFIRM_PHRASE}' to confirm")
        if confirm != LIVE_CONFIRM_PHRASE:
            print("Confirmation phrase did not match. Aborting.")
            sys.exit(1)
        print("Live mode confirmed.")

    # API credentials
    print("\n--- API credentials ---")
    print("  Required for demo/live. Leave blank for paper (no private stream).")
    api_key = _prompt("Bybit API key", secret=False)
    api_secret = _prompt("Bybit API secret", secret=True)

    if mode in ("demo", "live") and (not api_key or not api_secret):
        print("\nWARNING: Demo/live without API credentials will disable private stream,")
        print("portfolio refresh, and reconciliation. Order placement will not work.")
        if not _prompt_yes_no("Continue anyway?", default=False):
            sys.exit(1)

    # Order placement
    print("\n--- Order placement ---")
    placement_enabled = False
    if mode in ("demo", "live"):
        print("  Enabling order placement sends real orders to the exchange.")
        placement_enabled = _prompt_yes_no("Enable exchange order placement?", default=False)
        if placement_enabled:
            print("\n*** WARNING: ORDER PLACEMENT ENABLED ***")
            print("  Orders will be sent to the exchange.")
            _prompt("Press Enter to continue")
    dry_run = "false" if placement_enabled else "true"

    # Postgres
    print("\n--- Storage ---")
    use_postgres = _prompt_yes_no("Enable Postgres journaling?", default=False)
    postgres_dsn = ""
    if use_postgres:
        postgres_dsn = _prompt("PostgreSQL DSN (e.g. postgresql://user:pass@localhost:5432/trading)")

    # Parquet archive
    use_parquet = _prompt_yes_no("Enable Parquet archive output?", default=True)
    archive_dir = "data/archive"
    if use_parquet:
        archive_dir = _prompt("Archive directory", archive_dir)

    # Symbols
    print("\n--- Symbols ---")
    symbols_default = "BTCUSDT"
    symbols_raw = _prompt("Symbols (comma-separated, e.g. BTCUSDT,ETHUSDT)", symbols_default)
    try:
        symbols = _parse_symbols(symbols_raw)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    symbols_str = ",".join(symbols)

    # Logging
    log_level = _prompt("Log level", "INFO")
    log_json = "true" if _prompt_yes_no("JSON logging?", default=True) else "false"

    # Build .env
    env_name = MODE_TO_ENV.get(mode, "bybit_testnet")
    trading_mode = MODE_TO_TRADING_MODE.get(mode, mode)
    lines.append("# ML Bot — generated by bootstrap_env.py")
    lines.append("")
    lines.append("# Runtime")
    lines.append(f"TRADING_MODE={trading_mode}")
    lines.append("TRADING_CONFIG_DIR=configs")
    lines.append("TRADING_CONFIG_BASE=base.yaml")
    lines.append(f"TRADING_ENV={env_name}")
    lines.append(f"TRADING_DRY_RUN={dry_run}")
    lines.append(f"TRADING_SYMBOLS={symbols_str}")
    lines.append("")
    lines.append("# API (leave empty for paper)")
    lines.append(f"BYBIT_API_KEY={api_key}")
    lines.append(f"BYBIT_API_SECRET={api_secret}")
    lines.append("")
    lines.append("# Storage")
    if postgres_dsn:
        lines.append(f"TRADING_POSTGRES_DSN={postgres_dsn}")
    else:
        lines.append("# TRADING_POSTGRES_DSN=")
    lines.append(f"TRADING_ARCHIVE_DIR={archive_dir}")
    lines.append("")
    lines.append("# Logging")
    lines.append(f"TRADING_LOG_LEVEL={log_level}")
    lines.append(f"TRADING_LOG_JSON={log_json}")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    try:
        os.chmod(output_path, 0o600)
    except OSError:
        pass
    print(f"\nWrote {output_path}")


def run_non_interactive(
    *,
    output_path: Path,
    mode: str,
    env_name: str | None = None,
    api_key: str = "",
    api_secret: str = "",
    dry_run: bool = True,
    symbols: str = "BTCUSDT",
    postgres_dsn: str = "",
    archive_dir: str = "data/archive",
    log_level: str = "INFO",
    log_json: bool = True,
) -> None:
    """Non-interactive .env generation for automation."""
    env = env_name or MODE_TO_ENV.get(mode, "bybit_testnet")
    trading_mode = MODE_TO_TRADING_MODE.get(mode, mode)
    dry_run_str = "true" if dry_run else "false"
    log_json_str = "true" if log_json else "false"

    lines = [
        "# ML Bot — generated by bootstrap_env.py (non-interactive)",
        "",
        "# Runtime",
        f"TRADING_MODE={trading_mode}",
        "TRADING_CONFIG_DIR=configs",
        "TRADING_CONFIG_BASE=base.yaml",
        f"TRADING_ENV={env}",
        f"TRADING_DRY_RUN={dry_run_str}",
        f"TRADING_SYMBOLS={symbols}",
        "",
        "# API",
        f"BYBIT_API_KEY={api_key}",
        f"BYBIT_API_SECRET={api_secret}",
        "",
        "# Storage",
    ]
    if postgres_dsn:
        lines.append(f"TRADING_POSTGRES_DSN={postgres_dsn}")
    else:
        lines.append("# TRADING_POSTGRES_DSN=")
    lines.extend([
        f"TRADING_ARCHIVE_DIR={archive_dir}",
        "",
        "# Logging",
        f"TRADING_LOG_LEVEL={log_level}",
        f"TRADING_LOG_JSON={log_json_str}",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    try:
        os.chmod(output_path, 0o600)
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap .env for ML Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive (default):
    python scripts/bootstrap_env.py --output .env

  Non-interactive:
    python scripts/bootstrap_env.py --output .env --non-interactive --mode paper
    python scripts/bootstrap_env.py --output .env --non-interactive --mode demo --dry-run
        """,
    )
    parser.add_argument("--output", "-o", type=Path, default=Path(".env"), help="Output .env path")
    parser.add_argument("--install-path", type=Path, default=Path("."), help="Install/repo root path")
    parser.add_argument("--default-mode", default="paper", help="Default mode when interactive")
    parser.add_argument("--default-dry-run", type=lambda x: x.lower() == "true", default=True, help="Default dry_run")
    # Non-interactive
    parser.add_argument("--non-interactive", action="store_true", help="Skip prompts; use flags")
    parser.add_argument("--mode", choices=MODES, help="Runtime mode (non-interactive)")
    parser.add_argument("--env", dest="env_name", help="TRADING_ENV override (bybit_demo, bybit_testnet, bybit_mainnet)")
    parser.add_argument("--api-key", default="", help="Bybit API key (avoid in scripts)")
    parser.add_argument("--api-secret", default="", help="Bybit API secret (avoid in scripts)")
    parser.add_argument("--dry-run", type=lambda x: x.lower() in ("true", "1", "yes"), default=True, help="dry_run")
    parser.add_argument("--symbols", default="BTCUSDT", help="Comma-separated symbols")
    parser.add_argument("--postgres-dsn", default="", help="PostgreSQL DSN")
    parser.add_argument("--archive-dir", default="data/archive", help="Parquet archive directory")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--log-json", type=lambda x: x.lower() in ("true", "1", "yes"), default=True, help="JSON logs")
    args = parser.parse_args()

    if args.non_interactive:
        if not args.mode:
            parser.error("--mode required when --non-interactive")
        try:
            _parse_symbols(args.symbols)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        run_non_interactive(
            output_path=args.output,
            mode=args.mode,
            env_name=args.env_name,
            api_key=args.api_key,
            api_secret=args.api_secret,
            dry_run=args.dry_run,
            symbols=args.symbols,
            postgres_dsn=args.postgres_dsn,
            archive_dir=args.archive_dir,
            log_level=args.log_level,
            log_json=args.log_json,
        )
    else:
        run_interactive(
            output_path=args.output,
            install_path=args.install_path,
            default_mode=args.default_mode,
            default_dry_run=args.default_dry_run,
        )


if __name__ == "__main__":
    main()
