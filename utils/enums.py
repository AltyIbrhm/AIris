"""
Defines enums for signal types, trade sides, and other constants.
"""
from enum import Enum, auto

class SignalType(Enum):
    """Types of trading signals."""
    BUY = auto()
    SELL = auto()
    HOLD = auto()

class TradeSide(Enum):
    """Trading position sides."""
    LONG = auto()
    SHORT = auto()

class OrderType(Enum):
    """Types of trading orders."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()

class TimeFrame(Enum):
    """Trading timeframes."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

class StrategyType(Enum):
    """Types of trading strategies."""
    TECHNICAL = auto()
    AI = auto()
    HYBRID = auto() 