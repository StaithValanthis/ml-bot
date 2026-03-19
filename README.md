# ML Bot - Bybit USDT Perpetual Trading System

Production-grade modular monolith for Bybit USDT perpetual futures with strict execution correctness, risk controls, and backtest/paper/live parity.

## Objectives

- Deterministic and replayable trading core.
- Unified architecture across backtest, paper, demo/testnet, and live modes.
- Strong configuration validation, structured logging, and safe startup defaults.
- Extensible foundation for future ML retraining and controlled model promotion.

## Implemented Phases

- **Phase 1**: Project packaging, typed config (pydantic-settings + YAML), shared enums/dataclasses, structured logging, minimal async entrypoint.
- **Phase 2**: Bybit REST client with auth/signing, rate limiter, retry, request/response schemas.
- **Phase 3**: WebSocket clients (public/private), market data normalizers, market state, candle builder, staleness watchdog.
- **Phase 4**: Strategy (5m trend/breakout candidates), regime filter, signal engine, risk engine, sizing, circuit breaker, order manager, execution engine, reconciler.
- **Phase A**: REST envelope parsing fixes, unit tests for REST client.
- **Phase B**: Runtime orchestrator with task lifecycle, auth-conditional startup, graceful shutdown.
- **Phase C**: Observability (metrics, health, alerts) and durability (ledger, PnL tracker, Postgres/Parquet sinks).
- **Phase D**: WebSocket auth/subscribe ack validation, reconnect state reset, staleness expected channels, critical unit tests, cleanup.
- **Phase E**: Backtest foundation (event-driven engine, cost/funding models, walk-forward and optimizer scaffolds).
- **Phase F**: Config hardening (typed symbols, risk limits, model registry), research/model scaffolds (datasets, labeling, training, models).

## Planned Architecture Layers

- `config` (YAML + env merged at startup)
- `exchange` (Bybit REST/WS adapters)
- `marketdata` (state, candle building, staleness checks)
- `features` (feature definitions and computation)
- `models` (bundle, predictor, registry integration)
- `strategy` (candidate generation + regime filtering)
- `risk` (limits, circuit breakers, portfolio controls)
- `execution` (intent -> order lifecycle + reconciliation)
- `journal` (ledger and PnL accounting)
- `monitoring` (metrics, health, alerts)
- `storage` (Postgres/Timescale, parquet, cache)
- `research` (datasets, labeling, training, evaluation)
- `backtest` (event-driven replay + cost/funding models) — scaffold implemented
- `runtime` (mode orchestrator and scheduling)

## Run Modes

- `backtest`: offline deterministic event replay with synthetic or file-based event source. Use `TRADING_MODE=backtest` or `trading-backtest`.
- `paper`: live market data, simulated execution.
- `demo`: Bybit demo/testnet execution.
- `live`: real capital execution with full safeguards.

## Quick Start

1. Create a virtual environment and install:
   - `pip install -e ".[dev]"`
2. Copy `.env.example` to `.env` and set:
   - `TRADING_MODE` (`backtest`, `paper`, `demo`, or `live`)
   - `TRADING_ENV` (`bybit_mainnet`, `bybit_testnet`, or `bybit_demo`)
   - `BYBIT_API_KEY`, `BYBIT_API_SECRET` (required for demo/live execution)
3. Run:
   - **Runtime (paper/demo/live):** `python -m trading.main` or `trading-bot`
   - **Backtest:** `TRADING_MODE=backtest python -m trading.main` or `trading-backtest`

## Production Installation (Ubuntu 22.04/24.04)

The installer sets up Python, dependencies, configuration, and optional systemd services.

```bash
./install.sh
```

**Phases:**

1. **Prechecks** — OS compatibility, repo root, systemctl
2. **Python + venv** — apt packages, virtualenv, `pip install -e ".[dev]"`
3. **Directories** — `logs/`, `data/`, `data/archive/`
4. **Configuration** — Interactive prompts (`scripts/bootstrap_env.py`):
   - Mode: paper / demo / live / backtest-ready
   - API credentials (getpass for secret)
   - Order placement (dry_run)
   - Postgres, Parquet
   - Symbols
   - Log level / JSON
5. **Systemd** — Install mode-specific service (e.g. `trading-bot-paper.service`)

**Modes:**

- **paper** — Safest; no exchange orders. Good for first run.
- **demo** — Bybit demo (`bybit_demo` env, api-demo.bybit.com). Preferred over testnet.
- **live** — Real money. Requires typing `ENABLE LIVE TRADING` to confirm.

**Defaults:** paper mode, `dry_run=true`, BTCUSDT only, Parquet enabled.

**Service management:**

```bash
systemctl status trading-bot-paper.service   # or demo/live
systemctl start trading-bot-paper.service
systemctl stop trading-bot-paper.service
journalctl -u trading-bot-paper.service -f
```

**Troubleshooting:**

- `config file does not exist` — Ensure `configs/` and `TRADING_ENV` match (e.g. `bybit_demo.yaml`).
- `Symbol 'X' has no entry in symbols config` — Add symbol to `configs/symbols.yaml`.
- Service fails to start — Check `journalctl -u trading-bot-*.service -n 50`; verify `.env` and paths.

## Configuration Model

Settings are loaded from:

1. `configs/base.yaml`
2. `configs/{TRADING_ENV}.yaml`
3. `configs/symbols.yaml`
4. `configs/risk_limits.yaml`
5. `configs/model_registry.yaml` (optional; MLflow or similar)
6. `configs/logging.yaml`
7. Environment overrides (`TRADING_*`, plus Bybit secrets)

Optional env vars:

- `TRADING_DRY_RUN` — `true` (default) = no exchange orders; `false` = orders sent. Never silently set to `false` for demo/live.
- `TRADING_SYMBOLS` — Comma-separated symbols (e.g. `BTCUSDT,ETHUSDT`). Must exist in `configs/symbols.yaml`.
- `TRADING_POSTGRES_DSN` — PostgreSQL connection string for ledger durability (when unset, Postgres is skipped).
- `TRADING_ARCHIVE_DIR` — Parquet archive root (default: `data/archive`).

Startup is fail-fast: invalid config shapes or values terminate the process.

## Safety Baseline

- UTC-only timestamps enforced.
- Dry-run default enabled in base config.
- Strong typing for shared runtime/execution objects.
- No secrets hardcoded in repository.
- WebSocket auth and subscription acks validated before connection is considered healthy.
- Staleness watchdog trips circuit breaker and safe mode on feed health issues.

## Development Checks

- Run tests: `pytest`
- Lint: `ruff check .`
- Type-check: `mypy src`

## Demo / Testnet Execution

For supervised demo runs with exchange order placement, see [docs/DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md). It covers:
- Recommended supervised demo run procedure
- What to verify during first execution tests (ack timing, status transitions, reconciliation, stale-feed, circuit breaker)
- What is not implemented for full production readiness

For the first supervised live-like execution test, use [docs/FIRST_LIVE_CHECKLIST.md](docs/FIRST_LIVE_CHECKLIST.md): prerequisites, what to watch, abort triggers, and post-shutdown verification.

## Status

Phases 1–4, A–F are implemented. Backtest has a minimal CLI (`trading-backtest` or `TRADING_MODE=backtest`); file-based event loading is scaffolded. This is not yet production-ready for live capital.
