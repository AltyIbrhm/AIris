"""
Risk management and trade validation module.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json
import logging
import os
from utils.portfolio import PortfolioTracker

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config_path: str, portfolio_tracker: PortfolioTracker):
        """Initialize risk manager with configuration and portfolio tracker."""
        self.portfolio = portfolio_tracker
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.config_loaded = self.config is not None
        self.last_long_signals: Dict[str, Dict[str, Any]] = {}
        self.last_short_signals: Dict[str, Dict[str, Any]] = {}
        self.last_long_signal_time: Dict[str, datetime] = {}
        self.last_short_signal_time: Dict[str, datetime] = {}

    def _load_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """Load risk configuration from file."""
        try:
            if not os.path.exists(config_path):
                self.logger.warning(f"Config file {config_path} not found")
                return None

            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'default' not in config:
                    self.logger.warning("No default configuration found")
                    return None
                return config
        except Exception as e:
            self.logger.warning(f"Error loading risk config: {str(e)}")
            return None

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default risk configuration."""
        self.config_loaded = False
        return {
            'default': {
                'min_confidence': 0.3,
                'max_open_positions_total': 3,
                'max_open_positions_per_symbol': 1,
                'max_drawdown_percent': 10.0,
                'max_daily_loss': 300.0,
                'default_sl_percent': 2.0,
                'default_tp_percent': 4.0,
                'duplicate_signal_block_minutes': 5,
                'max_position_size_percent': 10.0,
                'max_leverage': 1.0,
                'risk_free_rate': 0.02,
                'volatility_lookback': 20,
                'position_sizing_method': 'kelly',
                'emergency_stop_loss_percent': 5.0
            }
        }

    def _get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for specific strategy, falling back to defaults."""
        if not strategy_name:
            return self.config.get('default', {})

        strategy_config = self.config.get(strategy_name, {})
        default_config = self.config.get('default', {})
        
        # Merge strategy config with defaults, strategy settings take precedence
        return {**default_config, **strategy_config}

    def check(self, signal: Dict[str, Any]) -> bool:
        """Check if signal passes risk management rules."""
        try:
            if not self.config_loaded:
                self.logger.warning("Risk config not loaded, rejecting signal")
                return False

            if not self._validate_signal(signal):
                return False

            strategy_config = self._get_strategy_config(signal.get('strategy', 'default'))
            
            # Check confidence
            if not self._check_confidence(signal, strategy_config):
                return False
                
            # Check position limits
            if not self._check_position_limits(signal, strategy_config):
                return False
                
            # Check drawdown
            if not self._check_drawdown(strategy_config):
                return False
                
            # Check daily loss
            if not self._check_daily_loss(strategy_config):
                return False
                
            # Check duplicate signals
            if not self._check_duplicate_signal(signal, strategy_config):
                return False
                
            # Update last signal tracking
            symbol = signal.get('symbol', 'default')
            direction = signal.get('direction', 'long')
            if direction == 'long':
                self.last_long_signals[symbol] = signal
                self.last_long_signal_time[symbol] = datetime.now()
            else:
                self.last_short_signals[symbol] = signal
                self.last_short_signal_time[symbol] = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in risk check: {str(e)}")
            return False

    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate basic signal structure."""
        required_fields = ['symbol', 'action', 'price', 'timestamp', 'confidence', 'strategy']
        if not all(field in signal for field in required_fields):
            self.logger.warning("Signal missing required fields")
            return False
        return True

    def _check_confidence(self, signal: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check if signal confidence meets minimum threshold."""
        min_confidence = config.get('min_confidence', 0.3)
        if signal['confidence'] < min_confidence:
            self.logger.warning(f"Signal confidence {signal['confidence']} below threshold {min_confidence}")
            return False
        return True

    def _check_position_limits(self, signal: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check if opening new position complies with limits."""
        max_positions_total = config.get('max_open_positions_total', 3)
        max_positions_per_symbol = config.get('max_open_positions_per_symbol', 1)
        
        # Check total positions
        total_positions = len(self.portfolio.get_open_positions())
        if total_positions >= max_positions_total:
            self.logger.warning(f"Total positions {total_positions} at limit {max_positions_total}")
            return False
            
        # Check positions per symbol
        symbol_positions = len(self.portfolio.get_open_positions(signal['symbol']).get(signal['symbol'], []))
        if symbol_positions >= max_positions_per_symbol:
            self.logger.warning(f"Symbol positions {symbol_positions} at limit {max_positions_per_symbol}")
            return False
            
        return True

    def _check_drawdown(self, config: Dict[str, Any]) -> bool:
        """Check if current drawdown is within limits."""
        max_drawdown = config.get('max_drawdown_percent', 10.0)
        current_drawdown = self.portfolio.get_current_drawdown()
        
        if current_drawdown >= max_drawdown:
            self.logger.warning(f"Current drawdown {current_drawdown:.2f}% exceeds limit {max_drawdown}%")
            return False
            
        return True

    def _check_daily_loss(self, config: Dict[str, Any]) -> bool:
        """Check if daily loss is within limits."""
        max_daily_loss = config.get('max_daily_loss', 300.0)
        daily_pnl = self.portfolio.get_daily_pnl()
        
        if daily_pnl < 0 and abs(daily_pnl) >= max_daily_loss:
            self.logger.warning(f"Daily loss {abs(daily_pnl)} exceeds limit {max_daily_loss}")
            return False
            
        return True

    def _check_duplicate_signal(self, signal: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check if signal is a duplicate of a recent signal."""
        symbol = signal.get('symbol', 'default')
        direction = signal.get('direction', 'long')
        block_minutes = config.get('duplicate_signal_block_minutes', 5)
        
        # Get the appropriate signal history based on direction
        last_signals = self.last_long_signals if direction == 'long' else self.last_short_signals
        last_signal_times = self.last_long_signal_time if direction == 'long' else self.last_short_signal_time
        
        # Check if we have a recent signal for this symbol and direction
        if symbol in last_signal_times:
            time_diff = datetime.now() - last_signal_times[symbol]
            if time_diff.total_seconds() < block_minutes * 60:
                last_signal = last_signals.get(symbol, {})
                
                # Check if price has changed significantly
                if abs(last_signal.get('price', 0) - signal.get('price', 0)) < 0.01:
                    self.logger.warning(f"Duplicate signal within {block_minutes} minutes")
                    return False
        
        return True

    def get_position_size(self, signal: Dict[str, Any], strategy_config: Dict[str, Any] = None) -> float:
        """Calculate position size based on risk parameters."""
        if strategy_config is None:
            strategy_config = self._get_strategy_config(signal.get('strategy', ''))
        
        price = signal['price']
        account_value = self.portfolio.current_capital
        max_position_size = strategy_config.get('max_position_size_percent', 10)
        
        # Calculate maximum position size based on account value
        max_size = account_value * (max_position_size / 100)
        
        # Adjust for volatility if available
        if 'volatility' in signal:
            volatility = signal['volatility']
            volatility_scale = strategy_config.get('volatility_scale', 1.0)
            max_size *= (volatility_scale / volatility)  # Reduce size for higher volatility
        
        # Apply strategy-specific position sizing method
        sizing_method = strategy_config.get('position_sizing_method', 'fixed')
        if sizing_method == 'kelly':
            win_rate = signal.get('win_rate', 0.5)
            risk_free_rate = strategy_config.get('risk_free_rate', 0.02)
            kelly_fraction = max(0, (win_rate - (1 - win_rate)) / (1 - risk_free_rate))
            max_size *= kelly_fraction
        
        return min(max_size, account_value * 0.1)  # Cap at 10% of account value 