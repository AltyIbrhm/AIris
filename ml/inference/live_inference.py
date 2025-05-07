import torch
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import pandas as pd
import ta

from ml.models.lstm_model import LSTMClassifier
from utils.logger import setup_logger
from utils.performance_tracker import PerformanceTracker
from strategy.exits.exit_manager import ExitManager
from strategy.filters.base_filter import BaseFilter
from market_data.base import BaseMarketDataFetcher
from utils.portfolio import PortfolioTracker
from utils.enums import TradeSide, SignalType

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = setup_logger(__name__, str(log_dir))

class LiveInference:
    def __init__(self, model_path: str):
        """Initialize LiveInference.
        
        Args:
            model_path: Path to saved model
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already added
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Load configs
        self.config = self._load_config()
        self.risk_config = self._load_risk_config()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize state
        self.last_prediction = None
        self.price_history = {
            "close": [],
            "high": [],
            "low": []
        }
        
        # Initialize components
        self.market_data = BaseMarketDataFetcher(self.config)
        self.portfolio = PortfolioTracker()
        self.performance_tracker = PerformanceTracker(self.config)
        
        # Initialize filters
        self.base_filter = BaseFilter(min_trade_spacing=self.config.get("trade_spacing", 5))
        self.current_candle_index = -1  # Start at -1 so first increment makes it 0
        
        # Initialize exit manager with risk parameters
        trailing_config = {
            "activation_threshold": self.risk_config["trailing_stop"]["activation_threshold"],
            "trail_distance_atr_mult": self.risk_config["trailing_stop"]["trailing_distance"],
            "max_loss_pct": self.risk_config["hard_stop"]["max_loss"],
            "max_holding_time": self.risk_config["position_sizing"]["max_holding_periods"]
        }
        
        hard_stop_config = {
            "stop_loss_atr_mult": self.risk_config["position_sizing"]["atr_multiplier"],
            "max_loss_pct": self.risk_config["hard_stop"]["max_loss"],
            "max_holding_time": self.risk_config["position_sizing"]["max_holding_periods"]
        }
        
        self.exit_manager = ExitManager(trailing_config, hard_stop_config)
        
        # Initialize state
        self.current_position = None
        self.ema_fast_values = []
        self.ema_slow_values = []
        self.rsi_values = []
        self.atr = None
        
        # Technical indicator parameters
        self.ema_fast = 8
        self.ema_slow = 21
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # Initialize lookback window
        self.lookback_window = self.config["data"]["lookback_window"]
        
    def _load_config(self) -> Dict:
        """Load the live configuration file."""
        with open("config/live_config.yaml", 'r') as f:
            return yaml.safe_load(f)
            
    def _load_risk_config(self) -> Dict:
        """Load the risk configuration file."""
        with open("config/risk_config.json", 'r') as f:
            return json.load(f)
            
    def _load_model(self, model_path: str) -> LSTMClassifier:
        """Load the trained model."""
        model = LSTMClassifier(
            input_dim=self.config["model"]["input_size"],
            hidden_dim=self.config["model"]["hidden_size"],
            num_layers=self.config["model"]["num_layers"]
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
        
    def update_price_history(self, high: float, low: float, close: float):
        """Update price history for technical indicators."""
        # Convert single values to arrays if needed
        if isinstance(high, (int, float)):
            high = np.array([high])
        if isinstance(low, (int, float)):
            low = np.array([low])
        if isinstance(close, (int, float)):
            close = np.array([close])
            
        # Convert to list if numpy array
        if isinstance(high, np.ndarray):
            high = high.tolist()
        if isinstance(low, np.ndarray):
            low = low.tolist()
        if isinstance(close, np.ndarray):
            close = close.tolist()
            
        # Append new values
        self.price_history["high"].extend(high)
        self.price_history["low"].extend(low)
        self.price_history["close"].extend(close)
        
        # Keep only necessary history
        max_lookback = max(self.lookback_window, self.ema_slow, self.rsi_period, 14)
        if len(self.price_history["close"]) > max_lookback:
            self.price_history["close"] = self.price_history["close"][-max_lookback:]
            self.price_history["high"] = self.price_history["high"][-max_lookback:]
            self.price_history["low"] = self.price_history["low"][-max_lookback:]
            
        # Update EMAs
        if len(self.price_history["close"]) >= self.ema_slow:
            closes = np.array(self.price_history["close"])
            self.ema_fast_values = self.calculate_ema(closes, self.ema_fast)
            self.ema_slow_values = self.calculate_ema(closes, self.ema_slow)
            
        # Update RSI
        if len(self.price_history["close"]) >= self.rsi_period:
            self.rsi_values = self.calculate_rsi(np.array(self.price_history["close"]))
            
        # Update ATR
        if len(self.price_history["high"]) >= 2:
            self.atr = self.calculate_atr(
                np.array(self.price_history["high"]),
                np.array(self.price_history["low"]),
                np.array(self.price_history["close"])
            )
        
    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average.
        
        Args:
            prices: Array of prices
            period: EMA period
            
        Returns:
            np.ndarray: EMA values
        """
        if len(prices) < period:
            return np.zeros_like(prices)
            
        # Calculate EMA
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
            
        return ema
        
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index.
        
        Args:
            prices: Array of prices
            period: RSI period
            
        Returns:
            np.ndarray: RSI values
        """
        if len(prices) < period + 1:
            return np.zeros_like(prices)
            
        # Calculate price changes
        deltas = np.diff(prices)
        deltas = np.append(deltas[0], deltas)  # Add first value to match length
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # First value
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        # Calculate subsequent values
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
        
        # Calculate RS and RSI
        rs = avg_gain / np.where(avg_loss == 0, 1e-6, avg_loss)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            period: ATR period
            
        Returns:
            np.ndarray: ATR values
        """
        if len(high) < 2:
            return np.zeros_like(high)
            
        # Calculate True Range
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Calculate ATR
        atr = np.zeros_like(high)
        atr[0] = tr[0]
        
        for i in range(1, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            
        return atr
        
    def is_trend_aligned(self, signal: SignalType) -> bool:
        """Check if signal aligns with EMA trend."""
        try:
            if len(self.price_history["close"]) < max(self.ema_fast, self.ema_slow):
                return True  # Not enough data to check trend
            
            # Calculate EMAs
            close_prices = np.array(self.price_history["close"])
            ema_fast = self.calculate_ema(close_prices, self.ema_fast)
            ema_slow = self.calculate_ema(close_prices, self.ema_slow)
            
            # Get latest values
            latest_fast = ema_fast[-1]
            latest_slow = ema_slow[-1]
            
            if signal == SignalType.LONG:
                return float(latest_fast) > float(latest_slow)  # Uptrend
            elif signal == SignalType.SHORT:
                return float(latest_fast) < float(latest_slow)  # Downtrend
            return True  # NEUTRAL signals always pass
            
        except Exception as e:
            self.logger.error(f"Error checking trend alignment: {str(e)}")
            return True  # Default to allowing trade on error
    
    def is_rsi_aligned(self, signal: SignalType) -> bool:
        """Check if signal aligns with RSI conditions."""
        try:
            if len(self.price_history["close"]) < self.rsi_period:
                return True  # Not enough data to calculate RSI
            
            # Calculate RSI
            close_prices = np.array(self.price_history["close"])
            rsi = self.calculate_rsi(close_prices, self.rsi_period)
            
            # Get latest RSI value
            latest_rsi = float(rsi[-1])
            
            if signal == SignalType.LONG:
                return latest_rsi < self.rsi_oversold  # Oversold condition
            elif signal == SignalType.SHORT:
                return latest_rsi > self.rsi_overbought  # Overbought condition
            return True  # NEUTRAL signals always pass
            
        except Exception as e:
            self.logger.error(f"Error checking RSI alignment: {str(e)}")
            return True  # Default to allowing trade on error

    def process_market_data(self, market_data: Dict) -> Optional[SignalType]:
        """Process incoming market data and generate trading signals."""
        try:
            # Update candle index
            self.current_candle_index += 1
            
            # Prepare features
            features = self._prepare_features(market_data)
            if features is None:
                return None
                
            # Get model prediction
            prediction = self.model(features)
            
            # Interpret prediction
            signal = self._interpret_prediction(prediction)
            
            # For non-neutral signals, check trade spacing
            if signal is not None and signal != SignalType.NEUTRAL:
                if not self.base_filter.check_trade_spacing(self.current_candle_index):
                    self.logger.info(f"Trade spacing check failed: current={self.current_candle_index}, last={self.base_filter.last_trade_index}")
                    return None
                    
                # Update last trade index only for non-neutral signals that pass spacing check
                self.base_filter.update_last_trade_index(self.current_candle_index)
                self.logger.info(f"Trade allowed at index {self.current_candle_index}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {str(e)}")
            return None
        
    def _prepare_features(self, market_data: Dict[str, Any]) -> torch.Tensor:
        """Prepare features for model input.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            torch.Tensor: Prepared features for model input
        """
        try:
            # Convert market data to pandas DataFrame for technical analysis
            df = pd.DataFrame({
                'open': market_data['open'],
                'close': market_data['close'],
                'high': market_data['high'],
                'low': market_data['low'],
                'volume': market_data['volume']
            })
            
            # Calculate technical indicators
            features = []
            
            # Price-based features (4 features)
            raw_close = df['close']  # Mean: 100.0, Std: 10.0
            returns = df['close'].pct_change().fillna(0)  # Mean: 0.0, Std: 0.01
            volume_price = df['volume'] * df['close']  # Mean: 100000.0, Std: 20000.0
            ema_ratio = df['close'] / df['close'].mean()  # Mean: 100.0, Std: 10.0
            features.extend([raw_close, returns, volume_price, ema_ratio])
            
            # EMAs (2 features)
            ema_fast = ta.trend.EMAIndicator(df['close'], window=self.ema_fast).ema_indicator().fillna(method='ffill')
            ema_slow = ta.trend.EMAIndicator(df['close'], window=self.ema_slow).ema_indicator().fillna(method='ffill')
            features.extend([ema_fast, ema_slow])  # EMAs: Mean: 50.0/2.0, Std: 20.0/0.5
            
            # RSI (1 feature)
            rsi = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period).rsi()
            rsi = rsi.fillna(50)  # Mean: 50.0, Std: 10.0
            features.append(rsi)
            
            # MACD (3 features)
            macd = ta.trend.MACD(df['close'])
            macd_line = macd.macd().fillna(0)  # Mean: 0.0, Std: 0.1
            signal_line = macd.macd_signal().fillna(0)  # Mean: 0.0, Std: 0.1
            macd_hist = macd.macd_diff().fillna(0)  # Mean: 0.0, Std: 0.1
            features.extend([macd_line, signal_line, macd_hist])
            
            # Bollinger Bands (3 features)
            bb = ta.volatility.BollingerBands(df['close'])
            bb_high = bb.bollinger_hband().fillna(method='ffill')
            bb_low = bb.bollinger_lband().fillna(method='ffill')
            bb_mavg = bb.bollinger_mavg().fillna(method='ffill')
            
            # Calculate BB width and position, handling potential division by zero
            bb_width = ((bb_high - bb_low) / bb_mavg).fillna(0)  # Mean: 4.0, Std: 1.0
            bb_position = ((df['close'] - bb_low) / (bb_high - bb_low + 1e-8)).fillna(0.5)  # Mean: 0.5, Std: 0.2
            bb_pct = ((df['close'] - bb_mavg) / (bb_high - bb_low + 1e-8)).fillna(0)  # Mean: 0.0, Std: 0.2
            features.extend([bb_width, bb_position, bb_pct])
            
            # Store raw values for testing (unscaled)
            raw_values = [f.values[-1] for f in features]
            feature_names = [
                'close', 'returns', 'volume_price', 'ema_ratio',
                'ema_fast', 'ema_slow', 'rsi',
                'macd_line', 'signal_line', 'macd_hist',
                'bb_width', 'bb_position', 'bb_pct'
            ]
            self.last_raw_features = dict(zip(feature_names, raw_values))
            
            # Combine features
            feature_matrix = np.column_stack([f.values for f in features])
            
            # Replace any remaining NaN or inf values with 0
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Take only the last row for prediction
            feature_matrix = feature_matrix[-1:]
            
            # Normalize features using config values
            feature_means = np.array(self.config["data"]["feature_mean"])
            feature_stds = np.array(self.config["data"]["feature_std"])
            
            # Ensure feature_matrix and normalization arrays have compatible shapes
            feature_matrix = feature_matrix.reshape(1, -1)
            feature_means = feature_means.reshape(1, -1)
            feature_stds = feature_stds.reshape(1, -1)
            
            # Apply normalization
            feature_matrix = (feature_matrix - feature_means) / feature_stds
            
            # Convert to tensor with correct shape
            tensor = torch.tensor(feature_matrix, dtype=torch.float32).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            raise
            
    def _interpret_prediction(self, prediction: torch.Tensor) -> SignalType:
        """Interpret model prediction and return trading signal.
        
        Args:
            prediction: Raw model prediction tensor
            
        Returns:
            SignalType: Trading signal (LONG, SHORT, or NEUTRAL)
        """
        try:
            # Ensure prediction is 2D
            if prediction.dim() == 1:
                prediction = prediction.unsqueeze(0)
            
            # Store prediction for position sizing
            self.last_prediction = prediction
            
            # Get highest probability and its index
            max_prob, max_idx = torch.max(prediction, dim=1)
            confidence = max_prob.item()
            
            # Check confidence threshold
            if confidence < self.config["inference"]["confidence_threshold"]:
                self.logger.info(f"Low confidence prediction ({confidence:.4f}), returning NEUTRAL")
                return SignalType.NEUTRAL
            
            # Map index to signal type (0: LONG, 1: SHORT, 2: NEUTRAL)
            signal_map = {
                0: SignalType.LONG,
                1: SignalType.SHORT,
                2: SignalType.NEUTRAL
            }
            return signal_map[max_idx.item()]
            
        except Exception as e:
            self.logger.error(f"Error interpreting prediction: {str(e)}")
            return SignalType.NEUTRAL
        
    def calculate_position_size(self, signal: SignalType) -> float:
        """Calculate position size based on risk parameters and model confidence.
        
        Args:
            signal: Trading signal (LONG or SHORT)
            
        Returns:
            float: Position size as a fraction of account value
        """
        try:
            # Get base risk per trade from config
            base_risk = self.risk_config["position_sizing"]["risk_per_trade"]
            min_conf = self.risk_config["position_sizing"]["min_confidence"]
            max_conf = self.risk_config["position_sizing"]["max_confidence"]
            
            # Get model confidence from last prediction
            if self.last_prediction is not None:
                max_prob, _ = torch.max(self.last_prediction, dim=1)
                confidence = max_prob.item()
                
                # Normalize confidence to [0, 1] range
                norm_conf = (confidence - min_conf) / (max_conf - min_conf)
                norm_conf = max(0.0, min(1.0, norm_conf))  # Clamp to [0, 1]
                
                # Scale position size linearly with confidence
                position_size = base_risk * (1.0 + norm_conf)
            else:
                confidence = min_conf
                position_size = base_risk
                
            # Apply min/max constraints
            min_size = self.risk_config["position_sizing"]["min_risk_per_trade"]
            max_size = self.risk_config["position_sizing"]["max_risk_per_trade"]
            position_size = max(min_size, min(max_size, position_size))
            
            self.logger.info(f"Calculated position size: {position_size:.4f} (confidence: {confidence:.4f})")
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return self.risk_config["position_sizing"]["min_risk_per_trade"]
        
    def calculate_stop_loss(self, entry_price: float, side: TradeSide) -> float:
        """Calculate stop loss level based on ATR."""
        if not self.atr:
            return 0.0
            
        stop_distance = self.atr * self.risk_config["position_sizing"]["atr_multiplier"]
        return entry_price - stop_distance if side == TradeSide.LONG else entry_price + stop_distance
        
    def calculate_take_profit(self, entry_price: float, side: TradeSide) -> float:
        """Calculate take profit level based on risk:reward ratio."""
        if not self.atr:
            return 0.0
            
        stop_distance = self.atr * self.risk_config["position_sizing"]["atr_multiplier"]
        reward_distance = stop_distance * self.risk_config["position_sizing"]["risk_reward_ratio"]
        
        return entry_price + reward_distance if side == TradeSide.LONG else entry_price - reward_distance
        
    def close_position(self, current_price: float):
        """Close the current position and update portfolio."""
        if not self.current_position:
            return
            
        pnl = self.portfolio.close_position(current_price)
        self.performance_tracker.update_metrics(pnl)
        self.current_position = None
        
    def run(self):
        """Main execution loop for live inference."""
        self.logger.info("Starting live inference...")
        
        try:
            while True:
                # Get latest market data
                market_data = self.market_data.get_latest_data()
                
                # Process market data and get signal
                signal = self.process_market_data(market_data)
                
                if signal:
                    # Check if we should exit current position
                    if self.current_position:
                        exit_signal = self.exit_manager.check_exit(
                            current_price=market_data["close"],
                            position=self.current_position
                        )
                        
                        if exit_signal:
                            self.close_position(market_data["close"])
                            self.logger.info(f"Position closed due to {exit_signal}")
                            
                    # Check if we should enter new position
                    elif signal != SignalType.NEUTRAL:
                        # Check trend and RSI alignment
                        if not self.is_trend_aligned(signal) or not self.is_rsi_aligned(signal):
                            self.logger.info("Signal rejected due to trend/RSI misalignment")
                            continue
                            
                        position_size = self.calculate_position_size(signal)
                        stop_loss = self.calculate_stop_loss(
                            market_data["close"],
                            TradeSide.LONG if signal == SignalType.LONG else TradeSide.SHORT
                        )
                        take_profit = self.calculate_take_profit(
                            market_data["close"],
                            TradeSide.LONG if signal == SignalType.LONG else TradeSide.SHORT
                        )
                        
                        self.current_position = self.portfolio.open_position(
                            side=TradeSide.LONG if signal == SignalType.LONG else TradeSide.SHORT,
                            size=position_size,
                            entry_price=market_data["close"],
                            stop_loss=stop_loss,
                            take_profit=take_profit
                        )
                        
                        # Initialize ExitManager for new position
                        self.exit_manager.initialize_trade(
                            entry_price=market_data["close"],
                            side='BUY' if signal == SignalType.LONG else 'SELL',
                            high_prices=self.price_history["high"],
                            low_prices=self.price_history["low"],
                            close_prices=self.price_history["close"],
                            position_size=position_size
                        )
                        
                        self.logger.info(f"New position opened: {self.current_position}")
                        
        except KeyboardInterrupt:
            self.logger.info("Live inference stopped by user")
        except Exception as e:
            self.logger.error(f"Error in live inference: {str(e)}")
        finally:
            # Close any open positions
            if self.current_position:
                self.close_position(market_data["close"])
                
            # Log final performance metrics
            self.performance_tracker.log_final_metrics()

def main():
    """Main entry point for live inference."""
    model_path = "ml/models/checkpoints/best_model.pth"
    live_inference = LiveInference(model_path)
    live_inference.run()

if __name__ == "__main__":
    main() 