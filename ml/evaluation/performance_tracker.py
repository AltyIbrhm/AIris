"""
Performance tracking for model evaluation.
"""
from typing import Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

class PerformanceTracker:
    """Tracks and reports trading performance metrics during evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize performance tracker with configuration."""
        self.config = config
        self.metrics = {}
        self.trades = {}
        
        # Create output directory
        self.output_dir = Path("ml/evaluation/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def update_metrics(self, symbol: str, trade_result: Dict[str, Any]) -> None:
        """Update metrics with a new trade result."""
        if symbol not in self.metrics:
            self.metrics[symbol] = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_trade_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'exit_reasons': {}
            }
            self.trades[symbol] = []
            
        # Add trade to history
        self.trades[symbol].append(trade_result)
        
        # Update basic metrics
        metrics = self.metrics[symbol]
        metrics['total_trades'] += 1
        metrics['total_pnl'] += trade_result['pnl']
        
        if trade_result['pnl'] > 0:
            metrics['winning_trades'] += 1
            
        # Update win rate
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
        
        # Update average trade PnL
        metrics['avg_trade_pnl'] = metrics['total_pnl'] / metrics['total_trades']
        
        # Update exit reasons distribution
        exit_reason = trade_result['exit_reason']
        metrics['exit_reasons'][exit_reason] = metrics['exit_reasons'].get(exit_reason, 0) + 1
        
        # Calculate drawdown
        if len(self.trades[symbol]) > 1:
            cumulative_pnl = np.cumsum([t['pnl'] for t in self.trades[symbol]])
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = (peak - cumulative_pnl) / peak
            metrics['max_drawdown'] = max(metrics['max_drawdown'], np.max(drawdown))
            
        # Calculate Sharpe ratio if we have enough trades
        if len(self.trades[symbol]) > 1:
            returns = np.array([t['pnl'] for t in self.trades[symbol]])
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                metrics['sharpe_ratio'] = avg_return / std_return * np.sqrt(252)  # Annualized
                
    def get_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get current metrics for a symbol."""
        return self.metrics.get(symbol, {})
        
    def export_metrics(self) -> None:
        """Export metrics and trade history to files."""
        # Export metrics to JSON
        metrics_file = self.output_dir / "performance_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        # Export trades to CSV
        for symbol in self.trades:
            trades_df = pd.DataFrame(self.trades[symbol])
            trades_file = self.output_dir / f"{symbol}_trades.csv"
            trades_df.to_csv(trades_file, index=False) 