"""
Risk management and trade validation module.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json
import logging
import os
from utils.portfolio import PortfolioTracker

class RiskManager:
    def __init__(self, config_path: str, portfolio_tracker: PortfolioTracker):
        """Initialize risk manager with configuration and portfolio tracker."""
        self.portfolio = portfolio_tracker
        self.logger = logging.getLogger(__name__)
        self.config_loaded = False
        self.config = self._load_config(config_path)
        self.last_signals: Dict[str, Dict[str, Any]] = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load risk configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'default' not in config:
                    self.logger.error("No default configuration found in risk config")
                    return {}
                self.config_loaded = True
                return config
        except Exception as e:
            self.logger.error(f"Error loading risk config: {str(e)}")
            self.config_loaded = False
            return {}

    def _get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for specific strategy, falling back to defaults."""
        if not strategy_name:
            return self.config.get('default', {})

        strategy_config = self.config.get(strategy_name, {})
        default_config = self.config.get('default', {})
        
        # Merge strategy config with defaults, strategy settings take precedence
        return {**default_config, **strategy_config}

    def check(self, signal: Dict[str, Any]) -> bool:
        """
        Check if a trade signal complies with risk parameters.
        Returns True if the trade is allowed, False otherwise.
        """
        try:
            # Reject all signals if config failed to load
            if not self.config_loaded:
                self.logger.error("Risk config not loaded, rejecting signal")
                return False

            # Get strategy-specific config
            strategy_name = signal.get('strategy', '')
            strategy_config = self._get_strategy_config(strategy_name)
            
            if not strategy_config:
                self.logger.error(f"No configuration found for strategy: {strategy_name}")
                return False

            # Basic signal validation
            if not self._validate_signal(signal):
                return False

            # Check confidence threshold
            if not self._check_confidence(signal, strategy_config):
                return False

            # Check position limits
            if not self._check_position_limits(signal, strategy_config):
                return False

            # Check drawdown limits
            if not self._check_drawdown(strategy_config):
                return False

            # Check daily loss limits
            if not self._check_daily_loss(strategy_config):
                return False

            # Check duplicate signals
            if not self._check_duplicate_signal(signal, strategy_config):
                return False

            # Check stop loss and take profit
            if not self._validate_sl_tp(signal, strategy_config):
                return False

            # Update last signal timestamp
            self._update_last_signal(signal)
            return True

        except Exception as e:
            self.logger.error(f"Error in risk check: {str(e)}")
            return False

    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate basic signal structure."""
        required_fields = ['action', 'price', 'timestamp', 'confidence', 'strategy']
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
        max_positions = config.get('max_open_positions', 1)
        current_positions = len(self.portfolio.get_open_positions())
        
        if current_positions >= max_positions:
            self.logger.warning(f"Max positions ({max_positions}) reached")
            return False
        return True

    def _check_drawdown(self, config: Dict[str, Any]) -> bool:
        """Check if current drawdown is within limits."""
        max_drawdown = config.get('max_drawdown_percent', 10)
        current_drawdown = self.portfolio.get_drawdown()
        
        if current_drawdown > max_drawdown:
            self.logger.warning(f"Drawdown {current_drawdown}% exceeds limit {max_drawdown}%")
            return False
        return True

    def _check_daily_loss(self, config: Dict[str, Any]) -> bool:
        """Check if daily loss is within limits."""
        max_daily_loss = config.get('max_daily_loss', 300)
        daily_pnl = self.portfolio.get_daily_pnl()
        
        if daily_pnl < -max_daily_loss:
            self.logger.warning(f"Daily loss {abs(daily_pnl)} exceeds limit {max_daily_loss}")
            return False
        return True

    def _check_duplicate_signal(self, signal: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check for duplicate signals within time window."""
        symbol = signal.get('symbol', 'default')
        block_minutes = config.get('duplicate_signal_block_minutes', 5)
        
        if symbol in self.last_signals:
            last_signal = self.last_signals[symbol]
            time_diff = datetime.now() - last_signal['timestamp']
            
            if (time_diff < timedelta(minutes=block_minutes) and 
                last_signal['action'] == signal['action']):
                self.logger.warning(f"Duplicate {signal['action']} signal for {symbol}")
                return False
        return True

    def _validate_sl_tp(self, signal: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Validate stop loss and take profit levels."""
        price = signal['price']
        default_sl = config.get('default_sl_percent', 2.0)
        default_tp = config.get('default_tp_percent', 4.0)
        
        # Calculate SL/TP levels
        if signal['action'] == 'buy':
            sl_price = price * (1 - default_sl/100)
            tp_price = price * (1 + default_tp/100)
        else:  # sell
            sl_price = price * (1 + default_sl/100)
            tp_price = price * (1 - default_tp/100)
        
        # Validate risk-reward ratio
        risk = abs(price - sl_price)
        reward = abs(tp_price - price)
        min_rr_ratio = config.get('min_rr_ratio', 1.5)  # Minimum risk-reward ratio
        
        if reward/risk < min_rr_ratio:
            self.logger.warning(f"Risk-reward ratio {reward/risk:.2f} below minimum {min_rr_ratio}")
            return False
            
        return True

    def _update_last_signal(self, signal: Dict[str, Any]) -> None:
        """Update last signal timestamp."""
        symbol = signal.get('symbol', 'default')
        self.last_signals[symbol] = {
            'action': signal['action'],
            'timestamp': datetime.now()
        }

    def get_position_size(self, signal: Dict[str, Any]) -> float:
        """Calculate position size based on risk parameters."""
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