# First Supervised Live-Like Execution Checklist

Use this checklist for the first supervised execution test with exchange order placement (demo/testnet). **Do not imply unattended production readiness.**

## Prerequisites

- [ ] Bybit demo/testnet API credentials configured
- [ ] `TRADING_MODE=demo` and `TRADING_ENV=bybit_testnet` (or equivalent)
- [ ] `dry_run: false` to enable order placement
- [ ] Risk limits set to small, test-sized positions only
- [ ] Terminal/session available for continuous monitoring

## What to Watch in Logs

| Log event | Meaning |
|-----------|---------|
| `execution_mode_banner` | Order placement enabled; confirm mode and dry_run |
| `order_submission_attempt` | Order sent to exchange |
| `order_ack_received` | Exchange acknowledged order |
| `order_state_transition` | Status change (e.g. New → Filled) |
| `reconcile_mismatch_detected` | Local and exchange state differ |
| `reconcile_ok` | Reconcile cycle found no issues |
| `feed_stale` | Market data stale; circuit breaker tripped |
| `circuit_breaker_trip` | Circuit breaker tripped |
| `session_summary_written` | Post-run report written |

## What Should Trigger an Abort

- **Unexpected task failure**: `runtime_task_failed` — stop and investigate
- **Repeated order rejections**: Multiple `order_submit_failed` — check credentials, limits, exchange status
- **Stale feed with no recovery**: `feed_stale` and no data resuming — stop and fix connectivity
- **Reconcile mismatch that persists**: `reconcile_mismatch_detected` every cycle with no resolution — investigate before continuing
- **Manual decision**: Any behavior that does not match expectations — stop and review

## What Must Be Verified After Shutdown

- [ ] `runtime_stop` in ledger; clean shutdown (no task crash)
- [ ] Session summary file written: `{archive_dir}/session_summaries/session_*.json`
- [ ] Session summary counts match expectations (decisions, intents, submissions, acks)
- [ ] Reconcile mismatch count and circuit breaker trips reviewed
- [ ] Exchange UI/API: open orders and positions match expectations
- [ ] No orphaned orders or unexpected positions

## Recovery Gaps (Not Implemented)

At startup and shutdown, `recovery_gaps` is logged. These are **not** implemented:

- Order state restore from durable store
- Auto-cancel of stray orders
- Auto-place of missing orders
- WS reconnect with order state merge

Supervised runs only. No unattended production readiness.
