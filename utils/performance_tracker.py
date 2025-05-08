"""
Performance tracking and reporting module.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import csv
import os
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class TradeMetrics:
    """Trade performance metrics."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    direction: str
    pnl: float
    pnl_percent: float
    duration: timedelta
    win: bool

@dataclass
class SymbolMetrics:
    """Per-symbol performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    avg_trade_duration: timedelta = timedelta()
    last_trade_time: Optional[datetime] = None
    trades: List[TradeMetrics] = None

    def __post_init__(self):
        if self.trades is None:
            self.trades = []

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        return (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0

    @property
    def avg_pnl(self) -> float:
        """Calculate average PnL per trade."""
        return self.total_pnl / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def avg_duration_seconds(self) -> float:
        """Calculate average trade duration in seconds."""
        if not self.trades:
            return 0.0
        total_seconds = sum(t.duration.total_seconds() for t in self.trades)
        return total_seconds / len(self.trades)

class PerformanceTracker:
    """Tracks and reports trading performance metrics."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize performance tracker with configuration."""
        self.config = config
        self.metrics: Dict[str, SymbolMetrics] = defaultdict(SymbolMetrics)
        self.global_metrics = SymbolMetrics()
        self.log_dir = config.get('log_dir', 'logs')
        self.csv_path = os.path.join(self.log_dir, 'performance_summary.csv')
        self.json_path = os.path.join(self.log_dir, 'performance_snapshot.json')
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize CSV if it doesn't exist
        if not os.path.exists(self.csv_path):
            self._initialize_csv()

    def _initialize_csv(self) -> None:
        """Initialize CSV file with headers."""
        headers = [
            'timestamp', 'symbol', 'total_trades', 'win_rate', 'total_pnl',
            'avg_pnl', 'max_drawdown', 'avg_duration_seconds', 'last_trade_time'
        ]
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def log_trade(self, trade: Dict[str, Any]) -> None:
        """Log a completed trade and update metrics."""
        try:
            # Create trade metrics
            trade_metrics = TradeMetrics(
                symbol=trade['symbol'],
                entry_time=trade['entry_time'],
                exit_time=trade['exit_time'],
                entry_price=trade['entry_price'],
                exit_price=trade['exit_price'],
                size=trade['size'],
                direction=trade['direction'],
                pnl=trade['pnl'],
                pnl_percent=trade['pnl_percent'],
                duration=trade['exit_time'] - trade['entry_time'],
                win=trade['pnl'] > 0
            )

            # Update symbol metrics
            symbol_metrics = self.metrics[trade['symbol']]
            symbol_metrics.total_trades += 1
            symbol_metrics.total_pnl += trade['pnl']
            if trade['pnl'] > 0:
                symbol_metrics.winning_trades += 1
            symbol_metrics.last_trade_time = trade['exit_time']
            symbol_metrics.trades.append(trade_metrics)

            # Update global metrics
            self.global_metrics.total_trades += 1
            self.global_metrics.total_pnl += trade['pnl']
            if trade['pnl'] > 0:
                self.global_metrics.winning_trades += 1
            self.global_metrics.last_trade_time = trade['exit_time']
            self.global_metrics.trades.append(trade_metrics)

            # Log trade
            logger.info(
                f"Trade logged for {trade['symbol']}: "
                f"PnL={trade['pnl']:.2f}, "
                f"Duration={trade_metrics.duration}"
            )

        except Exception as e:
            logger.error(f"Error logging trade: {str(e)}")

    def update_drawdown(self, symbol: str, equity: float) -> None:
        """Update drawdown metrics for a symbol."""
        try:
            metrics = self.metrics[symbol]
            metrics.current_equity = equity
            
            # Update peak equity
            if equity > metrics.peak_equity:
                metrics.peak_equity = equity
            
            # Calculate drawdown
            if metrics.peak_equity > 0:
                drawdown = ((metrics.peak_equity - equity) / metrics.peak_equity) * 100
                metrics.max_drawdown = max(metrics.max_drawdown, drawdown)
            
            logger.debug(f"Updated drawdown for {symbol}: {metrics.max_drawdown:.2f}%")

        except Exception as e:
            logger.error(f"Error updating drawdown for {symbol}: {str(e)}")

    def summarize(self) -> Dict[str, Any]:
        """Generate summary of all performance metrics."""
        try:
            summary = {
                'global': {
                    'total_trades': self.global_metrics.total_trades,
                    'win_rate': self.global_metrics.win_rate,
                    'total_pnl': self.global_metrics.total_pnl,
                    'avg_pnl': self.global_metrics.avg_pnl,
                    'max_drawdown': max(m.max_drawdown for m in self.metrics.values()),
                    'avg_duration_seconds': self.global_metrics.avg_duration_seconds,
                    'last_trade_time': self.global_metrics.last_trade_time.isoformat() 
                        if self.global_metrics.last_trade_time else None
                },
                'symbols': {}
            }

            # Add per-symbol metrics
            for symbol, metrics in self.metrics.items():
                summary['symbols'][symbol] = {
                    'total_trades': metrics.total_trades,
                    'win_rate': metrics.win_rate,
                    'total_pnl': metrics.total_pnl,
                    'avg_pnl': metrics.avg_pnl,
                    'max_drawdown': metrics.max_drawdown,
                    'avg_duration_seconds': metrics.avg_duration_seconds,
                    'last_trade_time': metrics.last_trade_time.isoformat() 
                        if metrics.last_trade_time else None
                }

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {}

    def export_summary(self) -> None:
        """Export performance summary to CSV and JSON."""
        try:
            summary = self.summarize()
            timestamp = datetime.now().isoformat()

            # Export to CSV
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'symbol', 'total_trades', 'win_rate', 'total_pnl',
                    'avg_pnl', 'max_drawdown', 'avg_duration_seconds', 'last_trade_time'
                ])
                
                # Write global metrics
                writer.writerow({
                    'timestamp': timestamp,
                    'symbol': 'GLOBAL',
                    'total_trades': summary['global']['total_trades'],
                    'win_rate': summary['global']['win_rate'],
                    'total_pnl': summary['global']['total_pnl'],
                    'avg_pnl': summary['global']['avg_pnl'],
                    'max_drawdown': summary['global']['max_drawdown'],
                    'avg_duration_seconds': summary['global']['avg_duration_seconds'],
                    'last_trade_time': summary['global']['last_trade_time']
                })
                
                # Write per-symbol metrics
                for symbol, metrics in summary['symbols'].items():
                    writer.writerow({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'total_trades': metrics['total_trades'],
                        'win_rate': metrics['win_rate'],
                        'total_pnl': metrics['total_pnl'],
                        'avg_pnl': metrics['avg_pnl'],
                        'max_drawdown': metrics['max_drawdown'],
                        'avg_duration_seconds': metrics['avg_duration_seconds'],
                        'last_trade_time': metrics['last_trade_time']
                    })

            # Export to JSON
            with open(self.json_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info("Performance summary exported successfully")

        except Exception as e:
            logger.error(f"Error exporting summary: {str(e)}")

    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get performance summary for a specific date."""
        if date is None:
            date = datetime.now().date()

        daily_metrics = defaultdict(lambda: {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'trades': []
        })

        # Initialize global metrics
        global_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'trades': []
        }

        # Aggregate trades for the specified date
        for symbol, metrics in self.metrics.items():
            for trade in metrics.trades:
                if trade.exit_time.date() == date:
                    # Update symbol metrics
                    daily_metrics[symbol]['total_trades'] += 1
                    daily_metrics[symbol]['total_pnl'] += trade.pnl
                    if trade.win:
                        daily_metrics[symbol]['winning_trades'] += 1
                    daily_metrics[symbol]['trades'].append(trade)

                    # Update global metrics
                    global_metrics['total_trades'] += 1
                    global_metrics['total_pnl'] += trade.pnl
                    if trade.win:
                        global_metrics['winning_trades'] += 1
                    global_metrics['trades'].append(trade)

        # Calculate summary metrics
        summary = {
            'date': date.isoformat(),
            'total_trades': global_metrics['total_trades'],
            'total_pnl': global_metrics['total_pnl'],
            'win_rate': (global_metrics['winning_trades'] / global_metrics['total_trades'] * 100)
                if global_metrics['total_trades'] > 0 else 0.0,
            'avg_pnl': global_metrics['total_pnl'] / global_metrics['total_trades']
                if global_metrics['total_trades'] > 0 else 0.0,
            'max_drawdown': global_metrics['max_drawdown'],
            'symbols': {}
        }

        for symbol, metrics in daily_metrics.items():
            summary['symbols'][symbol] = {
                'total_trades': metrics['total_trades'],
                'win_rate': (metrics['winning_trades'] / metrics['total_trades'] * 100)
                    if metrics['total_trades'] > 0 else 0.0,
                'total_pnl': metrics['total_pnl'],
                'avg_pnl': metrics['total_pnl'] / metrics['total_trades']
                    if metrics['total_trades'] > 0 else 0.0,
                'max_drawdown': metrics['max_drawdown']
            }

        return summary 