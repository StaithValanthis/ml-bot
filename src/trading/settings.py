from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from trading.util.types import ExchangeType, MarketSymbol, RuntimeMode


class EnvSettings(BaseSettings):
    """
    Environment-based configuration values.

    These values are intentionally narrow for Phase 1 and merged into YAML configs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRADING_",
        extra="ignore",
        case_sensitive=False,
    )

    mode: RuntimeMode = RuntimeMode.PAPER
    config_dir: Path = Path("configs")
    config_base: str = "base.yaml"
    env: str = "bybit_testnet"
    log_level: str | None = None
    log_json: bool | None = None


class SecretsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    bybit_api_key: SecretStr | None = Field(default=None, alias="BYBIT_API_KEY")
    bybit_api_secret: SecretStr | None = Field(default=None, alias="BYBIT_API_SECRET")


class DemoDrillSettings(BaseModel):
    """Demo execution drill config; DEMO-only, disabled by default."""

    enabled: bool = False
    symbol: str = "BTCUSDT"
    side: str = "Buy"
    qty: Decimal = Decimal("0.001")
    mode: str = "post_only"
    max_notional_usdt: Decimal = Decimal("100")

    @field_validator("side")
    @classmethod
    def _validate_side(cls, value: str) -> str:
        if value not in ("Buy", "Sell"):
            raise ValueError("side must be Buy or Sell")
        return value

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        if value not in ("post_only", "reduce_only"):
            raise ValueError("mode must be post_only or reduce_only")
        return value

    @field_validator("max_notional_usdt")
    @classmethod
    def _validate_max_notional(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("max_notional_usdt must be positive")
        return value


class RuntimeSettings(BaseModel):
    mode: RuntimeMode
    timezone: str = "UTC"
    shutdown_timeout_seconds: int = 20
    dry_run: bool = True
    backtest_bars: int = 1200
    backtest_fill_probability: float = 0.55
    backtest_fill_seed: int | None = 42
    demo_drill: DemoDrillSettings = Field(default_factory=DemoDrillSettings)
    model_filter_enabled: bool = False
    model_artifact_path: Path | None = None

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        if value != "UTC":
            raise ValueError("Only UTC timezone is allowed.")
        return value

    @field_validator("backtest_fill_probability")
    @classmethod
    def _validate_fill_probability(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("backtest_fill_probability must be between 0 and 1")
        return value


class ExchangeSettings(BaseModel):
    provider: ExchangeType = ExchangeType.BYBIT
    base_url: str
    public_ws_url: str
    private_ws_url: str
    testnet: bool
    recv_window_ms: int = 5000
    request_timeout_seconds: int = 15
    max_retries: int = 5
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    bybit_api_key: SecretStr | None = None
    bybit_api_secret: SecretStr | None = None

    @model_validator(mode="after")
    def _validate_credentials_by_mode(self) -> ExchangeSettings:
        if self.bybit_api_key is None or self.bybit_api_secret is None:
            # Allowed in backtest/paper mode. Runtime validation handled at orchestrator startup.
            return self
        if not self.bybit_api_key.get_secret_value() or not self.bybit_api_secret.get_secret_value():
            raise ValueError("BYBIT_API_KEY/BYBIT_API_SECRET cannot be empty strings.")
        return self


class TradingSettings(BaseModel):
    category: str = "linear"
    symbols: list[str]
    candle_timeframe: str = "5"
    regime_timeframe: str = "60"
    trade_only_closed_candles: bool = True

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one symbol must be configured.")
        if any(not symbol.endswith("USDT") for symbol in value):
            raise ValueError("Only USDT perpetual symbols are supported in v1.")
        return value


class SymbolSpec(BaseModel):
    """Per-symbol market metadata from symbols config."""

    qty_step: Decimal
    min_qty: Decimal
    price_tick: Decimal
    max_leverage: Decimal

    def to_market_symbol(self, symbol: str) -> MarketSymbol:
        return MarketSymbol(
            symbol=symbol,
            qty_step=self.qty_step,
            min_qty=self.min_qty,
            price_tick=self.price_tick,
            max_leverage=self.max_leverage,
        )


class PerSymbolRiskLimit(BaseModel):
    """Per-symbol risk limits from risk_limits config."""

    max_notional_usdt: Decimal
    max_position_abs: Decimal


class ModelRegistrySettings(BaseModel):
    """Model registry config for MLflow or similar."""

    provider: str = "mlflow"
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "bybit-usdt-perp"
    stage: str = "Staging"

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        allowed = {"mlflow", "none"}
        if value.lower() not in allowed:
            raise ValueError(f"provider must be one of {allowed}")
        return value.lower()


class RiskSettings(BaseModel):
    max_total_notional_usdt: Decimal
    max_leverage: Decimal
    daily_loss_limit_usdt: Decimal
    liquidation_buffer_bps: int
    safe_mode: bool = False
    per_symbol: dict[str, PerSymbolRiskLimit] = Field(default_factory=dict)


class LoggingSettings(BaseModel):
    level: str = "INFO"
    json_output: bool = Field(default=True, alias="json")
    include_timestamp: bool = True
    logger_name: str = "trading"


class AppSettings(BaseModel):
    runtime: RuntimeSettings
    exchange: ExchangeSettings
    trading: TradingSettings
    risk: RiskSettings
    logging: LoggingSettings
    symbols: dict[str, SymbolSpec] = Field(default_factory=dict)
    model_registry: ModelRegistrySettings | None = None

    @model_validator(mode="after")
    def _validate_symbols_match_trading(self) -> AppSettings:
        for sym in self.trading.symbols:
            if sym not in self.symbols:
                raise ValueError(
                    f"Symbol '{sym}' in trading.symbols has no entry in symbols config. "
                    "Add metadata for each traded symbol in configs/symbols.yaml."
                )
        return self

    def get_symbol_specs(self) -> dict[str, MarketSymbol]:
        """Return MarketSymbol map for all traded symbols. Valid after validation."""
        return {s: self.symbols[s].to_market_symbol(s) for s in self.trading.symbols}


def backtest_config_from_settings(settings: AppSettings) -> "BacktestConfig":
    """Build BacktestConfig from AppSettings for config-backed backtest runs."""
    from trading.backtest.engine import BacktestConfig
    from trading.risk.risk_engine import PerSymbolLimit

    per_symbol = {
        s: PerSymbolLimit(
            max_notional_usdt=p.max_notional_usdt,
            max_position_abs=p.max_position_abs,
        )
        for s, p in settings.risk.per_symbol.items()
    }
    return BacktestConfig(
        initial_equity_usdt=Decimal("10000"),
        candle_timeframe=settings.trading.candle_timeframe,
        regime_timeframe=settings.trading.regime_timeframe,
        max_total_notional_usdt=settings.risk.max_total_notional_usdt,
        max_leverage=settings.risk.max_leverage,
        daily_loss_limit_usdt=settings.risk.daily_loss_limit_usdt,
        liquidation_buffer_bps=settings.risk.liquidation_buffer_bps,
        symbol_specs=settings.get_symbol_specs(),
        per_symbol_limits=per_symbol,
        fill_probability=float(settings.runtime.backtest_fill_probability),
        fill_seed=settings.runtime.backtest_fill_seed,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML object: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def _load_config_stack(config_dir: Path, base_file: str, env_name: str) -> dict[str, Any]:
    base_cfg = _read_yaml(config_dir / base_file)
    env_cfg = _read_yaml(config_dir / f"{env_name}.yaml")
    symbols_cfg = _read_yaml(config_dir / "symbols.yaml")
    risk_limits_cfg = _read_yaml(config_dir / "risk_limits.yaml")
    logging_cfg = _read_yaml(config_dir / "logging.yaml")
    try:
        model_registry_cfg = _read_yaml(config_dir / "model_registry.yaml")
    except FileNotFoundError:
        model_registry_cfg = {}
    merged = _deep_merge(
        _deep_merge(_deep_merge(_deep_merge(base_cfg, env_cfg), symbols_cfg), risk_limits_cfg),
        logging_cfg,
    )
    return _deep_merge(merged, model_registry_cfg)


def load_settings() -> AppSettings:
    """
    Load and validate app settings from YAML + environment variables.

    Fail fast on any missing or invalid configuration.
    """
    env = EnvSettings()
    secrets = SecretsSettings()

    data = _load_config_stack(
        config_dir=env.config_dir,
        base_file=env.config_base,
        env_name=env.env,
    )

    # Runtime mode comes from environment to prevent accidental mode mismatch.
    runtime_cfg = data.setdefault("runtime", {})
    runtime_cfg["mode"] = env.mode.value
    if (dry_run_env := os.getenv("TRADING_DRY_RUN")) is not None:
        runtime_cfg["dry_run"] = dry_run_env.lower() in ("true", "1", "yes")
    if (symbols_env := os.getenv("TRADING_SYMBOLS")) is not None:
        data.setdefault("trading", {})["symbols"] = [
            s.strip().upper() for s in symbols_env.split(",") if s.strip()
        ]
    if "backtest_bars" not in runtime_cfg:
        try:
            runtime_cfg["backtest_bars"] = int(os.getenv("TRADING_BACKTEST_BARS", "1200"))
        except ValueError:
            runtime_cfg["backtest_bars"] = 1200
    if "backtest_fill_probability" not in runtime_cfg:
        try:
            runtime_cfg["backtest_fill_probability"] = float(
                os.getenv("TRADING_BACKTEST_FILL_PROBABILITY", "0.55")
            )
        except ValueError:
            runtime_cfg["backtest_fill_probability"] = 0.55
    if "backtest_fill_seed" not in runtime_cfg:
        seed_env = os.getenv("TRADING_BACKTEST_FILL_SEED")
        if seed_env is not None:
            try:
                runtime_cfg["backtest_fill_seed"] = int(seed_env)
            except ValueError:
                runtime_cfg["backtest_fill_seed"] = 42
        elif "backtest_fill_seed" not in runtime_cfg:
            runtime_cfg["backtest_fill_seed"] = 42

    demo_drill_cfg = runtime_cfg.setdefault("demo_drill", {})
    if (drill_enabled := os.getenv("TRADING_DEMO_DRILL_ENABLED")) is not None:
        demo_drill_cfg["enabled"] = drill_enabled.lower() in ("true", "1", "yes")
    if (drill_symbol := os.getenv("TRADING_DEMO_DRILL_SYMBOL")) is not None:
        demo_drill_cfg["symbol"] = drill_symbol.strip().upper()
    if (drill_side := os.getenv("TRADING_DEMO_DRILL_SIDE")) is not None:
        demo_drill_cfg["side"] = drill_side.strip()
    if (drill_qty := os.getenv("TRADING_DEMO_DRILL_QTY")) is not None:
        try:
            demo_drill_cfg["qty"] = Decimal(drill_qty.strip())
        except Exception:
            pass
    if (drill_mode := os.getenv("TRADING_DEMO_DRILL_MODE")) is not None:
        demo_drill_cfg["mode"] = drill_mode.strip().lower()
    if (drill_max_notional := os.getenv("TRADING_DEMO_DRILL_MAX_NOTIONAL_USDT")) is not None:
        try:
            demo_drill_cfg["max_notional_usdt"] = Decimal(drill_max_notional.strip())
        except Exception:
            pass

    if (model_filter := os.getenv("TRADING_MODEL_FILTER_ENABLED")) is not None:
        runtime_cfg["model_filter_enabled"] = model_filter.lower() in ("true", "1", "yes")
    if (model_path := os.getenv("TRADING_MODEL_ARTIFACT_PATH")) is not None:
        p = Path(model_path.strip())
        runtime_cfg["model_artifact_path"] = p if p else None

    if env.log_level is not None:
        data.setdefault("logging", {})["level"] = env.log_level
    if env.log_json is not None:
        data.setdefault("logging", {})["json"] = env.log_json

    data.setdefault("exchange", {})["bybit_api_key"] = secrets.bybit_api_key
    data.setdefault("exchange", {})["bybit_api_secret"] = secrets.bybit_api_secret

    try:
        return AppSettings.model_validate(data)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid configuration: {exc}") from exc
