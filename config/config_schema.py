"""
Pydantic-based schema for validating runtime configuration.
"""
from pydantic import BaseModel, Field
from typing import Optional

class APIConfig(BaseModel):
    key: str
    secret: str
    base_url: str = "https://api.binance.us"

class TradingConfig(BaseModel):
    pair: str
    interval: str
    max_position_size: float
    max_drawdown: float = Field(ge=0, le=1)

class RiskConfig(BaseModel):
    stop_loss_pct: float = Field(ge=0, le=1)
    take_profit_pct: float = Field(ge=0, le=1)
    max_daily_trades: int = Field(ge=1)

class LoggingConfig(BaseModel):
    level: str
    file: str
    max_size: int
    backup_count: int

class Config(BaseModel):
    api: APIConfig
    trading: TradingConfig
    risk: RiskConfig
    logging: LoggingConfig 