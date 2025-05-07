import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import torch
from ml.data.dataset_builder import TimeSeriesDataset
from ml.models.lstm_model import LSTMClassifier

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
        position_size: float = 0.02,  # Reduced to 2% of balance per trade
        atr_multiplier: float = 1.2,  # Tightened ATR multiplier for SL/TP
        max_holding_periods: int = 24,  # Maximum holding periods (hours)
        min_holding_periods: int = 3,   # Minimum holding periods (bars)
        confidence_threshold: float = 0.85,  # Increased confidence threshold
        max_loss_pct: float = 0.01,  # 1% maximum loss per trade
        ema_fast: int = 8,  # Fast EMA period
        ema_slow: int = 21,  # Slow EMA period
        rsi_period: int = 14,  # RSI period
        rsi_overbought: int = 70,  # RSI overbought threshold
        rsi_oversold: int = 30,  # RSI oversold threshold
        trailing_stop_activation: float = 0.5,  # Activate trailing stop at 50% of target
        trailing_stop_distance: float = 0.8  # 80% of ATR for trailing stop
    ):
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.atr_multiplier = atr_multiplier
        self.max_holding_periods = max_holding_periods
        self.min_holding_periods = min_holding_periods
        self.confidence_threshold = confidence_threshold
        self.max_loss_pct = max_loss_pct
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.trailing_stop_activation = trailing_stop_activation
        self.trailing_stop_distance = trailing_stop_distance
        
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
        signal: int,
        price: float,
        timestamp: datetime,
        confidence: float,
        high: float,
        low: float,
        close: float
    ) -> None:
        """Execute a trade based on signal and confidence"""
        # Update price history for indicators
        self.update_price_history(high, low, close)
        
        # Skip if confidence is too low
        if confidence < self.confidence_threshold:
            return
            
        # Skip HOLD signals
        if signal == 1:
            return
            
        # Check trend alignment
        if not self.is_trend_aligned(signal):
            return
            
        # Check RSI alignment
        if not self.is_rsi_aligned(signal):
            return
            
        atr = self.calculate_atr()
        if atr == 0:
            return
            
        if self.current_position is None:  # No position, can enter
            position_size = self.calculate_position_size(self.balance_history[-1], price, atr)
            if position_size == 0:
                return
                
            stop_loss = price - (atr * self.atr_multiplier) if signal == 0 else price + (atr * self.atr_multiplier)
            take_profit = price + (atr * self.atr_multiplier) if signal == 0 else price - (atr * self.atr_multiplier)
            
            self.current_position = {
                'type': 'SELL' if signal == 0 else 'BUY',
                'entry_price': price,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': timestamp,
                'entry_balance': self.balance_history[-1],
                'holding_bars': 0,  # Track minimum holding period
                'entry_trend': 'uptrend' if self.ema_fast_values[-1] > self.ema_slow_values[-1] else 'downtrend',
                'entry_rsi': self.rsi_values[-1] if self.rsi_values else None
            }
            
        else:  # Have position, check exit conditions
            # Update holding period
            self.current_position['holding_bars'] += 1
            
            # Update trailing stop if conditions are met
            self.update_trailing_stop(price, atr)
            
            # Calculate current PnL
            current_pnl = (price - self.current_position['entry_price']) * self.current_position['position_size']
            if self.current_position['type'] == 'BUY':
                current_pnl = -current_pnl
                
            # Check maximum loss limit
            max_loss = self.balance_history[-1] * self.max_loss_pct
            if current_pnl < -max_loss:
                self.close_position(price, timestamp, 'max_loss')
                return
                
            # Skip exit checks if minimum holding period not met
            if self.current_position['holding_bars'] < self.min_holding_periods:
                return
                
            # Check stop loss and take profit
            if (self.current_position['type'] == 'SELL' and price >= self.current_position['stop_loss']) or \
               (self.current_position['type'] == 'BUY' and price <= self.current_position['stop_loss']):
                self.close_position(price, timestamp, 'stop_loss')
            elif (self.current_position['type'] == 'SELL' and price <= self.current_position['take_profit']) or \
                 (self.current_position['type'] == 'BUY' and price >= self.current_position['take_profit']):
                self.close_position(price, timestamp, 'take_profit')
    
    def close_position(self, price: float, timestamp: datetime, reason: str) -> None:
        """Close current position and record trade"""
        if self.current_position is None:
            return
            
        holding_periods = (timestamp - self.current_position['entry_time']).total_seconds() / 3600  # Convert to hours
        pnl = (price - self.current_position['entry_price']) * self.current_position['position_size']
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
        """Calculate trading performance metrics"""
        if not self.trades:
            return {}
            
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        
        # PnL metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        max_drawdown = 0
        peak = self.initial_balance
        for balance in self.balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0)
        returns = pd.Series(self.balance_history).pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        
        # Additional metrics
        avg_holding_period = trades_df['holding_periods'].mean()
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        # Trend analysis
        trend_aligned_trades = len(trades_df[trades_df['entry_trend'] == trades_df['exit_trend']])
        trend_alignment_rate = trend_aligned_trades / total_trades if total_trades > 0 else 0
        
        # RSI analysis
        rsi_aligned_trades = len(trades_df[
            ((trades_df['type'] == 'BUY') & (trades_df['entry_rsi'] < self.rsi_oversold)) |
            ((trades_df['type'] == 'SELL') & (trades_df['entry_rsi'] > self.rsi_overbought))
        ])
        rsi_alignment_rate = rsi_aligned_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': self.balance_history[-1],
            'return_pct': (self.balance_history[-1] - self.initial_balance) / self.initial_balance * 100,
            'avg_holding_period': avg_holding_period,
            'exit_reasons': exit_reasons,
            'trend_alignment_rate': trend_alignment_rate,
            'rsi_alignment_rate': rsi_alignment_rate
        }
    
    def plot_results(self, save_path: Path) -> None:
        """Plot equity curve and drawdown"""
        plt.figure(figsize=(15, 10))
        
        # Equity curve
        plt.subplot(2, 1, 1)
        plt.plot(self.balance_history, label='Equity Curve')
        plt.title('Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Balance')
        plt.grid(True)
        plt.legend()
        
        # Drawdown
        plt.subplot(2, 1, 2)
        peak = pd.Series(self.balance_history).expanding().max()
        drawdown = (peak - self.balance_history) / peak * 100
        plt.plot(drawdown, label='Drawdown %')
        plt.title('Drawdown')
        plt.xlabel('Trade Number')
        plt.ylabel('Drawdown %')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def load_config() -> Dict:
    """Load model configuration"""
    with open("config/model_config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded successfully")
    
    # Initialize dataset
    logger.info("Loading test dataset...")
    dataset = TimeSeriesDataset(
        features_path=config["features_path"],
        labels_path=config["labels_path"],
        seq_len=config["seq_len"]
    )
    
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
    
    # Initialize simulator with improved risk parameters and filters
    simulator = PnLSimulator(
        initial_balance=10000.0,
        position_size=0.02,  # 2% position size
        atr_multiplier=1.2,  # Tighter ATR multiplier
        max_holding_periods=24,
        min_holding_periods=3,  # Minimum 3 bars holding
        confidence_threshold=0.85,  # Higher confidence threshold
        max_loss_pct=0.01,  # 1% max loss per trade
        ema_fast=8,  # Fast EMA period
        ema_slow=21,  # Slow EMA period
        rsi_period=14,  # RSI period
        rsi_overbought=70,  # RSI overbought threshold
        rsi_oversold=30,  # RSI oversold threshold
        trailing_stop_activation=0.5,  # Activate trailing stop at 50% of target
        trailing_stop_distance=0.8  # 80% of ATR for trailing stop
    )
    
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
            simulator.execute_trade(
                prediction.item(),
                price,
                dataset.timestamps[i],
                confidence.item(),
                high,
                low,
                close
            )
    
    # Calculate and log metrics
    metrics = simulator.calculate_metrics()
    logger.info("\n📊 Trading Performance Metrics:")
    for key, value in metrics.items():
        if key != 'exit_reasons':
            logger.info(f"{key}: {value:.2f}")
    logger.info("\nExit Reasons Distribution:")
    for reason, count in metrics['exit_reasons'].items():
        logger.info(f"{reason}: {count}")
    
    # Plot results
    output_dir = Path("ml/evaluation/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    simulator.plot_results(output_dir / "pnl_simulation.png")
    logger.info(f"Results plotted and saved to {output_dir}/pnl_simulation.png")
    
    # Save trade history
    trades_df = pd.DataFrame(simulator.trades)
    trades_df.to_csv(output_dir / "trade_history.csv", index=False)
    logger.info(f"Trade history saved to {output_dir}/trade_history.csv")

if __name__ == "__main__":
    main() 