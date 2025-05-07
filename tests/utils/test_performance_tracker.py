"""
Tests for performance tracking and reporting.
"""
import pytest
from datetime import datetime, timedelta
import json
import os
from utils.performance_tracker import PerformanceTracker, TradeMetrics, SymbolMetrics

@pytest.fixture
def performance_config():
    """Create test performance configuration."""
    return {
        'log_dir': 'test_logs',
        'export_interval': 3600,
        'export_formats': ['csv', 'json'],
        'metrics': {
            'track_win_rate': True,
            'track_pnl': True,
            'track_drawdown': True,
            'track_duration': True,
            'track_daily_summary': True
        }
    }

@pytest.fixture
def sample_trade():
    """Create a sample trade for testing."""
    entry_time = datetime.now() - timedelta(hours=1)
    exit_time = datetime.now()
    return {
        'symbol': 'BTC/USDT',
        'entry_time': entry_time,
        'exit_time': exit_time,
        'entry_price': 50000.0,
        'exit_price': 51000.0,
        'size': 0.1,
        'direction': 'long',
        'pnl': 100.0,
        'pnl_percent': 2.0
    }

@pytest.fixture
def performance_tracker(performance_config):
    """Create a performance tracker instance."""
    tracker = PerformanceTracker(performance_config)
    yield tracker
    # Cleanup
    if os.path.exists('test_logs'):
        for file in os.listdir('test_logs'):
            os.remove(os.path.join('test_logs', file))
        os.rmdir('test_logs')

def test_log_trade(performance_tracker, sample_trade):
    """Test logging a trade."""
    performance_tracker.log_trade(sample_trade)
    
    # Check symbol metrics
    metrics = performance_tracker.metrics[sample_trade['symbol']]
    assert metrics.total_trades == 1
    assert metrics.winning_trades == 1
    assert metrics.total_pnl == 100.0
    assert metrics.last_trade_time == sample_trade['exit_time']
    
    # Check global metrics
    assert performance_tracker.global_metrics.total_trades == 1
    assert performance_tracker.global_metrics.winning_trades == 1
    assert performance_tracker.global_metrics.total_pnl == 100.0

def test_update_drawdown(performance_tracker):
    """Test drawdown tracking."""
    symbol = 'BTC/USDT'
    
    # Initial equity
    performance_tracker.update_drawdown(symbol, 10000.0)
    assert performance_tracker.metrics[symbol].peak_equity == 10000.0
    assert performance_tracker.metrics[symbol].max_drawdown == 0.0
    
    # Drop in equity
    performance_tracker.update_drawdown(symbol, 9000.0)
    assert performance_tracker.metrics[symbol].max_drawdown == 10.0
    
    # New peak
    performance_tracker.update_drawdown(symbol, 11000.0)
    assert performance_tracker.metrics[symbol].peak_equity == 11000.0
    
    # Larger drop
    performance_tracker.update_drawdown(symbol, 9000.0)
    assert abs(performance_tracker.metrics[symbol].max_drawdown - 18.18) < 0.01  # Allow small floating-point differences

def test_summarize(performance_tracker, sample_trade):
    """Test summary generation."""
    # Add some trades
    performance_tracker.log_trade(sample_trade)
    performance_tracker.log_trade({
        **sample_trade,
        'symbol': 'ETH/USDT',
        'pnl': -50.0,
        'pnl_percent': -1.0
    })
    
    summary = performance_tracker.summarize()
    
    # Check global metrics
    assert summary['global']['total_trades'] == 2
    assert summary['global']['win_rate'] == 50.0
    assert summary['global']['total_pnl'] == 50.0
    assert summary['global']['avg_pnl'] == 25.0
    
    # Check symbol metrics
    assert summary['symbols']['BTC/USDT']['total_trades'] == 1
    assert summary['symbols']['BTC/USDT']['win_rate'] == 100.0
    assert summary['symbols']['BTC/USDT']['total_pnl'] == 100.0
    
    assert summary['symbols']['ETH/USDT']['total_trades'] == 1
    assert summary['symbols']['ETH/USDT']['win_rate'] == 0.0
    assert summary['symbols']['ETH/USDT']['total_pnl'] == -50.0

def test_export_summary(performance_tracker, sample_trade):
    """Test summary export to CSV and JSON."""
    # Add a trade
    performance_tracker.log_trade(sample_trade)
    
    # Export summary
    performance_tracker.export_summary()
    
    # Check CSV file
    assert os.path.exists(performance_tracker.csv_path)
    with open(performance_tracker.csv_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 3  # Header + global + symbol
    
    # Check JSON file
    assert os.path.exists(performance_tracker.json_path)
    with open(performance_tracker.json_path, 'r') as f:
        data = json.load(f)
        assert 'global' in data
        assert 'symbols' in data
        assert 'BTC/USDT' in data['symbols']

def test_daily_summary(performance_tracker, sample_trade):
    """Test daily summary generation."""
    # Add trades for different days
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    
    # Today's trade
    performance_tracker.log_trade(sample_trade)
    
    # Yesterday's trade
    performance_tracker.log_trade({
        **sample_trade,
        'entry_time': yesterday,
        'exit_time': yesterday + timedelta(hours=1),
        'pnl': -50.0
    })
    
    # Get today's summary
    today_summary = performance_tracker.get_daily_summary(today.date())
    assert today_summary['symbols']['BTC/USDT']['total_trades'] == 1
    assert today_summary['symbols']['BTC/USDT']['total_pnl'] == 100.0
    
    # Get yesterday's summary
    yesterday_summary = performance_tracker.get_daily_summary(yesterday.date())
    assert yesterday_summary['symbols']['BTC/USDT']['total_trades'] == 1
    assert yesterday_summary['symbols']['BTC/USDT']['total_pnl'] == -50.0

def test_error_handling(performance_tracker):
    """Test error handling in performance tracker."""
    # Test invalid trade data
    invalid_trade = {'symbol': 'BTC/USDT'}  # Missing required fields
    performance_tracker.log_trade(invalid_trade)
    
    # Metrics should remain unchanged
    assert performance_tracker.global_metrics.total_trades == 0
    
    # Test invalid drawdown update
    performance_tracker.update_drawdown('BTC/USDT', 'invalid')  # Invalid equity value
    
    # Test export with no data
    performance_tracker.export_summary()
    assert os.path.exists(performance_tracker.csv_path)
    assert os.path.exists(performance_tracker.json_path) 