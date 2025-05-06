"""
Shared utility functions used across the project.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_pnl(entry_price: float, exit_price: float, size: float, side: str) -> float:
    """Calculate profit/loss for a trade."""
    if side == 'long':
        return (exit_price - entry_price) * size
    else:  # short
        return (entry_price - exit_price) * size

def calculate_drawdown(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown from an equity curve."""
    peak = equity_curve[0]
    max_dd = 0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        max_dd = max(max_dd, dd)
    
    return max_dd

def format_timestamp(timestamp: int) -> str:
    """Format Unix timestamp to readable string."""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate configuration dictionary has all required keys."""
    return all(key in config for key in required_keys)

def calculate_position_size(account_balance: float, risk_per_trade: float, 
                          entry_price: float, stop_loss: float) -> float:
    """Calculate position size based on risk management rules."""
    risk_amount = account_balance * risk_per_trade
    price_risk = abs(entry_price - stop_loss)
    return risk_amount / price_risk if price_risk > 0 else 0 