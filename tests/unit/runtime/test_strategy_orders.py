"""Unit tests for strategy order outcome tracking."""

from __future__ import annotations

from trading.runtime.strategy_orders import StrategyOrderOutcomes


def test_strategy_order_outcomes_defaults() -> None:
    """StrategyOrderOutcomes has correct defaults."""
    o = StrategyOrderOutcomes()
    assert o.intents == 0
    assert o.submissions == 0
    assert o.acks == 0
    assert o.filled == 0
    assert o.cancelled == 0
    assert o.rejected == 0
    assert o.partially_filled == 0


def test_strategy_order_outcomes_tracks_classification_counts() -> None:
    """StrategyOrderOutcomes tracks intents, submissions, acks, filled, cancelled, rejected, partially_filled."""
    o = StrategyOrderOutcomes()
    o.intents = 3
    o.submissions = 2
    o.acks = 2
    o.partially_filled = 1
    o.filled = 1
    o.cancelled = 0
    o.rejected = 0
    assert o.intents == 3
    assert o.submissions == 2
    assert o.acks == 2
    assert o.partially_filled == 1
    assert o.filled == 1
