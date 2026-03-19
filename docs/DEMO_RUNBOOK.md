# Supervised Demo Run Procedure

This runbook describes how to run and validate the trading bot in demo/testnet mode with exchange order placement enabled.

## Prerequisites

- Bybit demo/testnet API credentials (`BYBIT_API_KEY`, `BYBIT_API_SECRET`)
- `TRADING_ENV=bybit_testnet` or `bybit_demo` in `.env`
- `TRADING_MODE=demo` for testnet execution
- `dry_run: false` in config (or equivalent env override) to enable order placement

## Recommended Supervised Demo Run Procedure

1. **Pre-flight**
   - Confirm you are on testnet/demo, not mainnet
   - Verify `TRADING_MODE=demo` and `dry_run=false`
   - Ensure only small, test-sized positions are possible (check risk limits)

2. **Start the runtime**
   ```bash
   TRADING_MODE=demo python -m trading.main
   ```
   - Watch for the startup banner: `*** EXCHANGE ORDER PLACEMENT ENABLED ***`
   - Confirm `execution_mode_warning` and `execution_mode_banner` in logs

3. **Monitor during run**
   - Keep a terminal/session open to observe structured logs
   - Watch for `order_submission_attempt`, `order_ack_received`, `order_state_transition`
   - Check reconcile cycle logs every ~30 seconds

4. **Stop**
   - Use SIGINT (Ctrl+C) or the configured shutdown mechanism
   - Confirm `runtime_stop` in ledger and clean shutdown

## What to Verify During First Execution Tests

### Ack timing
- After `order_submission_attempt`, expect `order_ack_received` within a few seconds
- If no ack, check network, API credentials, and exchange status

### Order status transitions
- `order_state_transition` events should show `from_status` → `to_status` (e.g. New → PartiallyFilled → Filled)
- Verify transitions match exchange UI or API responses

### Reconciliation results
- `reconcile_ok` when local and exchange state match
- `reconcile_mismatch_detected` when there are issues; check `issue_types` and `reconcile_recovery_action`
- Note: auto-cancel and auto-place are **not** implemented; local state is synced from exchange only

### Stale-feed behavior
- If market data stops, expect `feed_stale` alert and `staleness_violation` in ledger
- Circuit breaker should trip; safe mode enabled
- No new orders placed while stale

### Circuit breaker behavior
- Order rejections increment toward threshold; at threshold, circuit breaker trips
- `circuit_breaker_trip` in ledger; `circuit_breaker_tripped` in health snapshot
- No new orders until cooldown expires

## What Is Not Implemented for Full Production Readiness

- **Order state recovery**: Order state is in-memory only; no restore from durable store on restart
- **Auto-recovery from reconcile**: No auto-cancel of stray orders, no auto-place of missing orders
- **Reduce-only exits in decision path**: `build_exit_intent` exists but is not wired into the decision loop
- **Stop orders**: `build_stop_intent` is scaffold only; not routed to exchange
- **Full position/wallet reconciliation**: Reconcile covers orders and basic reduce-only checks; no wallet sync
- **Prometheus/telemetry export**: Metrics are in-memory; no external scrape endpoint
- **Graceful reconnect with order state merge**: WS reconnect does not merge prior order state
