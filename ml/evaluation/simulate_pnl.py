import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import torch
from ml.data.dataset_builder import TimeSeriesDataset
from ml.models.lstm_model import LSTMClassifier
import json
from ml.evaluation.performance_tracker import PerformanceTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PnLSimulator:
    def __init__(
        self,
        initial_balance: float = 10000.0,
        position_size: float = None,  # Will be loaded from config
        atr_multiplier: float = None,  # Will be loaded from config
        max_holding_periods: int = None,  # Will be loaded from config
        min_holding_periods: int = 3,
        confidence_threshold: float = None,  # Will be loaded from config
        max_loss_pct: float = None,  # Will be loaded from config
        ema_fast: int = 8,
        ema_slow: int = 21,
        rsi_period: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        trailing_stop_activation: float = None,  # Will be loaded from config
        trailing_stop_distance: float = None  # Will be loaded from config
    ):
        # Load risk config
        with open("config/risk_config.json", "r") as f:
            risk_config = json.load(f)
            
        # Initialize with config values or defaults
        self.initial_balance = initial_balance
        self.position_size = position_size or float(risk_config.get('max_position_size_percent', 0.02))
        self.atr_multiplier = atr_multiplier or float(risk_config.get('atr_multiplier', 1.2))
        self.max_holding_periods = max_holding_periods or int(risk_config.get('max_holding_time', 24))
        self.min_holding_periods = min_holding_periods
        self.confidence_threshold = confidence_threshold or float(risk_config.get('min_confidence', 0.85))
        self.max_loss_pct = max_loss_pct or float(risk_config.get('max_loss_pct', 0.01))
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.trailing_stop_activation = trailing_stop_activation or float(risk_config.get('trailing_stop_activation', 0.5))
        self.trailing_stop_distance = trailing_stop_distance or float(risk_config.get('trailing_stop_distance', 0.8))
        
        # Performance tracking
        self.balance_history = [initial_balance]
        self.trades: List[Dict] = []
        self.current_position = None
        
        # Price history for ATR, EMA, and RSI calculation
        self.price_history = []
        self.ema_fast_values = []
        self.ema_slow_values = []
        self.rsi_values = []
        
    def update_price_history(self, high: float, low: float, close: float):
        """Update price history for ATR, EMA, and RSI calculation"""
        self.price_history.append({
            'high': high,
            'low': low,
            'close': close
        })
        
        # Keep only necessary history
        max_period = max(self.ema_slow, self.rsi_period, 14)
        if len(self.price_history) > max_period:
            self.price_history.pop(0)
            
        # Update EMAs
        if len(self.price_history) >= self.ema_slow:
            closes = [p['close'] for p in self.price_history]
            self.ema_fast_values = self.calculate_ema(closes, self.ema_fast)
            self.ema_slow_values = self.calculate_ema(closes, self.ema_slow)
            
        # Update RSI
        if len(self.price_history) >= self.rsi_period:
            self.rsi_values = self.calculate_rsi([p['close'] for p in self.price_history])
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return []
            
        multiplier = 2 / (period + 1)
        ema = [prices[0]]  # Initialize with first price
        
        for price in prices[1:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
            
        return ema
    
    def calculate_rsi(self, prices: List[float]) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < self.rsi_period + 1:
            return []
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:self.rsi_period])
        avg_loss = np.mean(losses[:self.rsi_period])
        
        if avg_loss == 0:
            return [100.0]
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return [rsi]
    
    def is_trend_aligned(self, signal: int) -> bool:
        """Check if signal aligns with current trend"""
        if not self.ema_fast_values or not self.ema_slow_values:
            return False
            
        fast_ema = self.ema_fast_values[-1]
        slow_ema = self.ema_slow_values[-1]
        
        if signal == 2:  # BUY
            return fast_ema > slow_ema  # Uptrend
        elif signal == 0:  # SELL
            return fast_ema < slow_ema  # Downtrend
        return True  # HOLD signals always pass
    
    def is_rsi_aligned(self, signal: int) -> bool:
        """Check if RSI aligns with signal"""
        if not self.rsi_values:
            return True
            
        rsi = self.rsi_values[-1]
        
        if signal == 2:  # BUY
            return rsi < self.rsi_oversold  # Oversold condition
        elif signal == 0:  # SELL
            return rsi > self.rsi_overbought  # Overbought condition
        return True  # HOLD signals always pass
    
    def calculate_atr(self) -> float:
        """Calculate Average True Range (ATR)"""
        if len(self.price_history) < 2:
            return 0.0
            
        tr_values = []
        for i in range(1, len(self.price_history)):
            high = self.price_history[i]['high']
            low = self.price_history[i]['low']
            prev_close = self.price_history[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)
            
        return np.mean(tr_values)
    
    def calculate_position_size(self, balance: float, price: float, atr: float) -> float:
        """Calculate position size based on ATR and risk management"""
        if atr == 0:
            return 0.0
        risk_amount = balance * self.position_size
        position_size = risk_amount / (atr * self.atr_multiplier)
        return position_size
    
    def update_trailing_stop(self, price: float, atr: float) -> None:
        """Update trailing stop if conditions are met"""
        if self.current_position is None:
            return
            
        # Calculate profit target
        target = abs(self.current_position['take_profit'] - self.current_position['entry_price'])
        current_profit = abs(price - self.current_position['entry_price'])
        
        # Check if trailing stop should be activated
        if current_profit >= target * self.trailing_stop_activation:
            trailing_distance = atr * self.trailing_stop_distance
            
            if self.current_position['type'] == 'SELL':
                new_stop = price + trailing_distance
                if new_stop < self.current_position['stop_loss']:
                    self.current_position['stop_loss'] = new_stop
            else:  # BUY
                new_stop = price - trailing_distance
                if new_stop > self.current_position['stop_loss']:
                    self.current_position['stop_loss'] = new_stop
    
    def execute_trade(
        self,
        signal: Dict[str, Any],  # Changed from int to Dict
        price: float,
        high: float,
        low: float,
        close: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Execute a trade based on the signal and current market conditions
        
        Args:
            signal (Dict[str, Any]): Trading signal dictionary containing:
                - action: str ("buy", "sell", or "hold")
                - confidence: float (model confidence)
                - price: float (signal price)
            price (float): Current market price
            high (float): Current bar's high price
            low (float): Current bar's low price
            close (float): Current bar's close price
            timestamp (Optional[datetime]): Current timestamp (default: current time)
        """
        # Use current time if timestamp not provided
        timestamp = timestamp or datetime.now()
        
        # Update technical indicators
        self.update_price_history(high, low, close)
        
        # Check if we should close existing position
        if self.current_position:
            # Update trailing stop if active
            atr = self.calculate_atr()
            self.update_trailing_stop(price, atr)
            
            # Check stop conditions
            stop_hit = price <= self.current_position['stop_loss'] if self.current_position['type'] == 'BUY' \
                else price >= self.current_position['stop_loss']
                
            take_profit_hit = price >= self.current_position['take_profit'] if self.current_position['type'] == 'BUY' \
                else price <= self.current_position['take_profit']
                
            # Close position if any exit condition is met
            if stop_hit:
                self.close_position(price, timestamp, "stop_loss")
                return
            elif take_profit_hit:
                self.close_position(price, timestamp, "take_profit")
                return
                
            # Check time-based exit
            holding_time = timestamp - self.current_position['entry_time']
            if holding_time.total_seconds() / 3600 >= self.max_holding_periods:
                self.close_position(price, timestamp, "time_exit")
                return
        
        # Only open new position if confidence meets threshold
        if signal['confidence'] < self.confidence_threshold:
            return
            
        # Convert signal action to internal format
        signal_type = 2 if signal['action'] == 'buy' else 0 if signal['action'] == 'sell' else 1
            
        # Check if signal aligns with trend and RSI
        if not self.is_trend_aligned(signal_type) or not self.is_rsi_aligned(signal_type):
            return
            
        # Calculate ATR for position sizing and stop placement
        atr = self.calculate_atr()
        if atr == 0:
            return
            
        # Open new position based on signal
        if signal['action'] == 'buy' and not self.current_position:
            stop_loss = price - (atr * self.atr_multiplier)
            take_profit = price + (atr * self.atr_multiplier * 2)  # 2:1 reward-risk
            
            position_size = self.calculate_position_size(self.balance_history[-1], price, atr)
            
            self.current_position = {
                'type': 'BUY',
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'size': position_size,
                'entry_time': timestamp
            }
            
        elif signal['action'] == 'sell' and not self.current_position:
            stop_loss = price + (atr * self.atr_multiplier)
            take_profit = price - (atr * self.atr_multiplier * 2)  # 2:1 reward-risk
            
            position_size = self.calculate_position_size(self.balance_history[-1], price, atr)
            
            self.current_position = {
                'type': 'SELL',
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'size': position_size,
                'entry_time': timestamp
            }
    
    def close_position(self, price: float, timestamp: datetime, reason: str) -> None:
        """Close current position and record trade"""
        if self.current_position is None:
            return
            
        holding_periods = (timestamp - self.current_position['entry_time']).total_seconds() / 3600  # Convert to hours
        pnl = (price - self.current_position['entry_price']) * self.current_position['size']
        if self.current_position['type'] == 'BUY':
            pnl = -pnl
            
        # Apply maximum loss limit if needed
        max_loss = self.balance_history[-1] * self.max_loss_pct
        if pnl < -max_loss:
            pnl = -max_loss
            
        trade_record = {
            **self.current_position,
            'exit_price': price,
            'exit_time': timestamp,
            'holding_periods': holding_periods,
            'pnl': pnl,
            'exit_reason': reason,
            'exit_balance': self.balance_history[-1] + pnl,
            'exit_trend': 'uptrend' if self.ema_fast_values[-1] > self.ema_slow_values[-1] else 'downtrend',
            'exit_rsi': self.rsi_values[-1] if self.rsi_values else None
        }
        
        self.trades.append(trade_record)
        self.balance_history.append(trade_record['exit_balance'])
        self.current_position = None
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate trading performance metrics
        
        Returns:
            Dict: Dictionary containing performance metrics
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "equity_curve": self.balance_history,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0
            }
            
        # Calculate basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit metrics
        gross_profit = sum([t['pnl'] for t in self.trades if t['pnl'] > 0])
        gross_loss = abs(sum([t['pnl'] for t in self.trades if t['pnl'] < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate drawdown
        peak = self.initial_balance
        max_drawdown = 0.0
        for balance in self.balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        returns = np.diff(self.balance_history) / self.balance_history[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "equity_curve": self.balance_history,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio
        }
    
    def plot_results(self) -> None:
        """Generate performance plots"""
        # Create figures directory if it doesn't exist
        figures_dir = Path("ml/evaluation/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(self.balance_history)
        plt.title("Equity Curve")
        plt.xlabel("Trade Number")
        plt.ylabel("Account Balance")
        plt.grid(True)
        plt.savefig(figures_dir / "equity_curve.png")
        plt.close()
        
        # Plot profit distribution if there are trades
        if self.trades:
            profits = [trade['pnl'] for trade in self.trades]
            plt.figure(figsize=(12, 6))
            plt.hist(profits, bins=50)
            plt.title("Profit Distribution")
            plt.xlabel("Profit/Loss")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(figures_dir / "profit_distribution.png")
            plt.close()

def load_config() -> Dict:
    """Load model configuration"""
    with open("config/model_config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded successfully")
    
    # Initialize dataset and performance tracker
    logger.info("Loading test dataset...")
    dataset = TimeSeriesDataset(
        features_path=config["features_path"],
        labels_path=config["labels_path"],
        seq_len=config["seq_len"]
    )
    
    performance_tracker = PerformanceTracker(config)
    
    # Load model
    logger.info("Loading trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(
        input_dim=dataset[0][0].shape[1],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(device)
    
    checkpoint = torch.load(config["save_path"], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded successfully")
    
    # Initialize simulator
    simulator = PnLSimulator()
    
    # Run simulation
    logger.info("Starting PnL simulation...")
    with torch.no_grad():
        for i in range(len(dataset)):
            features, label = dataset[i]
            features = features.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get model prediction
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            
            # Get price data
            price = features[0, -1, 0].item()  # Assuming first feature is price
            high = features[0, -1, 1].item()   # Assuming second feature is high
            low = features[0, -1, 2].item()    # Assuming third feature is low
            close = features[0, -1, 3].item()  # Assuming fourth feature is close
            
            # Execute trade
            signal = {
                'action': 'buy' if prediction.item() == 2 else 'sell' if prediction.item() == 0 else 'hold',
                'confidence': confidence.item(),
                'price': price,
                'symbol': 'BTCUSDT'  # Add symbol for performance tracking
            }
            
            simulator.execute_trade(
                signal,
                price,
                high,
                low,
                close,
                dataset.timestamps[i]
            )
            
            # Update performance metrics
            if simulator.trades and simulator.trades[-1]['exit_time'] == dataset.timestamps[i]:
                last_trade = simulator.trades[-1]
                performance_tracker.update_metrics(
                    symbol=signal['symbol'],
                    trade_result={
                        'entry_price': last_trade['entry_price'],
                        'exit_price': last_trade['exit_price'],
                        'entry_time': last_trade['entry_time'],
                        'exit_time': last_trade['exit_time'],
                        'pnl': last_trade['pnl'],
                        'exit_reason': last_trade['exit_reason'],
                        'position_size': last_trade['size']
                    }
                )
    
    # Calculate and log metrics using performance tracker
    metrics = performance_tracker.get_metrics('BTCUSDT')
    logger.info("\n📊 Trading Performance Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")
    
    # Plot results
    simulator.plot_results()
    
    # Save trade history
    trades_df = pd.DataFrame(simulator.trades)
    trades_df.to_csv(Path("ml/evaluation/figures/trade_history.csv"), index=False)
    logger.info(f"Trade history saved to ml/evaluation/figures/trade_history.csv")
    
    # Export performance metrics
    performance_tracker.export_metrics()

def simulate_trading(predictions: np.ndarray, probabilities: np.ndarray, prices: pd.DataFrame) -> Dict:
    """
    Simulate trading based on model predictions and probabilities
    
    Args:
        predictions (np.ndarray): Model predictions (0: SELL, 1: HOLD, 2: BUY)
        probabilities (np.ndarray): Prediction probabilities for each class
        prices (pd.DataFrame): Price data with high, low, close columns
        
    Returns:
        Dict: Simulation metrics including:
            - total_trades: Number of trades executed
            - win_rate: Percentage of winning trades
            - equity_curve: List of account balance values
            - max_drawdown: Maximum drawdown percentage
            - profit_factor: Gross profit / Gross loss
            - sharpe_ratio: Risk-adjusted return metric
    """
    # Initialize simulator with default settings
    simulator = PnLSimulator()
    
    # Convert prices to DataFrame if it's a numpy array
    if isinstance(prices, np.ndarray):
        if len(prices.shape) == 1 or prices.shape[1] == 1:
            # If 1D array or single column, use same values for high, low, close
            prices = pd.DataFrame({
                'high': prices.flatten(),
                'low': prices.flatten(),
                'close': prices.flatten()
            })
        else:
            # If 2D array, assume columns are high, low, close
            prices = pd.DataFrame({
                'high': prices[:, 0],
                'low': prices[:, 1],
                'close': prices[:, 2]
            })
    
    # Ensure we have a datetime index
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.date_range(start='2023-01-01', periods=len(prices), freq='1H')
    
    # Run simulation for each prediction
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        if i >= len(prices):
            break
            
        price_data = prices.iloc[i]
        simulator.execute_trade(
            {
                'action': 'buy' if pred == 2 else 'sell' if pred == 0 else 'hold',
                'confidence': np.max(prob),
                'price': price_data['close'],
                'symbol': 'BTCUSDT'
            },
            price_data['close'],
            price_data['high'],
            price_data['low'],
            price_data['close'],
            prices.index[i].to_pydatetime()  # Convert pandas timestamp to datetime
        )
    
    # Generate plots
    simulator.plot_results()
        
    # Calculate and return metrics
    return simulator.calculate_metrics()

if __name__ == "__main__":
    main() 