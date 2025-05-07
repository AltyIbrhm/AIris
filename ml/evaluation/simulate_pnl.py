import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PnLSimulator:
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize the PnL simulator.
        
        Args:
            initial_capital (float): Initial capital for trading
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []  # List of (entry_price, position_size, direction)
        self.trades = []  # List of (entry_price, exit_price, position_size, direction, pnl)
        self.equity_curve = [initial_capital]
        
    def simulate(self, 
                predictions: np.ndarray,
                probabilities: np.ndarray,
                prices: np.ndarray,
                confidence_threshold: float = 0.7) -> Dict:
        """
        Simulate trading based on model predictions.
        
        Args:
            predictions: Array of predicted labels (0: SELL, 1: HOLD, 2: BUY)
            probabilities: Array of prediction probabilities
            prices: Array of price data
            confidence_threshold: Minimum probability threshold for taking trades
            
        Returns:
            Dictionary containing performance metrics
        """
        for i in range(len(predictions)):
            current_price = prices[i]
            pred = predictions[i]
            prob = probabilities[i]
            
            # Close existing positions if prediction is opposite
            self._close_positions(current_price, pred)
            
            # Open new positions if confidence is high enough
            if prob[pred] >= confidence_threshold:
                if pred == 2:  # BUY
                    self._open_position(current_price, 1.0, "LONG")
                elif pred == 0:  # SELL
                    self._open_position(current_price, 1.0, "SHORT")
            
            # Update equity curve
            self.equity_curve.append(self._calculate_current_equity(current_price))
        
        return self._calculate_metrics()
    
    def _open_position(self, price: float, size: float, direction: str):
        """Open a new trading position."""
        self.positions.append((price, size, direction))
        logger.debug(f"Opened {direction} position at {price}")
    
    def _close_positions(self, current_price: float, prediction: int):
        """Close existing positions if prediction suggests opposite action."""
        for pos in self.positions[:]:
            entry_price, size, direction = pos
            should_close = (
                (direction == "LONG" and prediction == 0) or  # Close long on SELL
                (direction == "SHORT" and prediction == 2)    # Close short on BUY
            )
            
            if should_close:
                pnl = self._calculate_pnl(entry_price, current_price, size, direction)
                self.trades.append((entry_price, current_price, size, direction, pnl))
                self.current_capital += pnl
                self.positions.remove(pos)
                logger.debug(f"Closed {direction} position: Entry={entry_price}, Exit={current_price}, PnL={pnl:.2f}")
    
    def _calculate_pnl(self, entry: float, exit: float, size: float, direction: str) -> float:
        """Calculate PnL for a trade."""
        if direction == "LONG":
            return size * (exit - entry)
        else:  # SHORT
            return size * (entry - exit)
    
    def _calculate_current_equity(self, current_price: float) -> float:
        """Calculate current equity including open positions."""
        equity = self.current_capital
        for entry_price, size, direction in self.positions:
            equity += self._calculate_pnl(entry_price, current_price, size, direction)
        return equity
    
    def _calculate_metrics(self) -> Dict:
        """Calculate trading performance metrics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "equity_curve": self.equity_curve
            }
        
        # Calculate basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t[4] > 0)
        win_rate = winning_trades / total_trades
        
        # Calculate profit metrics
        profits = [t[4] for t in self.trades]
        avg_profit = np.mean(profits)
        
        # Calculate drawdown
        equity_curve = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Calculate Sharpe ratio (assuming daily returns)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if len(returns) > 0 else 0
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "equity_curve": self.equity_curve
        }

def plot_performance(metrics: Dict, save_dir: Path):
    """Plot trading performance metrics."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(metrics["equity_curve"])
    plt.title("Equity Curve")
    plt.xlabel("Trading Period")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.savefig(save_dir / "equity_curve.png")
    plt.close()
    
    # Plot profit distribution
    if metrics["total_trades"] > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(metrics["profits"], bins=50)
        plt.title("Profit Distribution")
        plt.xlabel("Profit/Loss")
        plt.ylabel("Frequency")
        plt.savefig(save_dir / "profit_distribution.png")
        plt.close()

def simulate_trading(predictions: np.ndarray,
                    probabilities: np.ndarray,
                    prices: np.ndarray,
                    initial_capital: float = 10000.0,
                    confidence_threshold: float = 0.7) -> Dict:
    """
    Run trading simulation and return performance metrics.
    
    Args:
        predictions: Array of predicted labels
        probabilities: Array of prediction probabilities
        prices: Array of price data
        initial_capital: Initial trading capital
        confidence_threshold: Minimum probability threshold for trades
        
    Returns:
        Dictionary containing performance metrics
    """
    simulator = PnLSimulator(initial_capital)
    metrics = simulator.simulate(predictions, probabilities, prices, confidence_threshold)
    
    # Plot performance metrics
    plot_performance(metrics, Path("ml/evaluation/figures"))
    
    # Log results
    logger.info("\n📈 Trading Performance Metrics:")
    logger.info(f"Total Trades: {metrics['total_trades']}")
    logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
    logger.info(f"Average Profit per Trade: ${metrics['avg_profit']:.2f}")
    logger.info(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    return metrics

if __name__ == "__main__":
    # Example usage
    from ml.evaluation.evaluate_model import evaluate
    
    # Get predictions and probabilities from model evaluation
    eval_results = evaluate()
    
    # Load price data (you'll need to implement this based on your data structure)
    # prices = load_price_data()
    
    # Run simulation
    # metrics = simulate_trading(
    #     eval_results["predictions"],
    #     eval_results["probabilities"],
    #     prices
    # ) 