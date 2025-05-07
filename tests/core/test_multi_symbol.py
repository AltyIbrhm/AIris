"""
Tests for multi-symbol trading functionality.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from core.loop import TradingLoop
from utils.portfolio import PortfolioTracker
from risk.checker import RiskManager

@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return {
        'exchange': {
            'name': 'binance',
            'api_key': 'test_key',
            'api_secret': 'test_secret'
        },
        'trading': {
            'exchange': 'binance',
            'symbols': ['ETH/USDT', 'BTC/USDT', 'SOL/USDT'],
            'interval': '5m',
            'poll_interval': 10,
            'risk_config': {
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
            },
            'ai_model_config': {
                'model_type': 'lstm',
                'input_features': ['open', 'high', 'low', 'close', 'volume'],
                'output_features': ['direction', 'confidence'],
                'sequence_length': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 100,
                'validation_split': 0.2
            },
            'paper_trading': True,
            'log_level': 'INFO'
        },
        'risk': {
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
        },
        'model': {
            'model_type': 'lstm',
            'input_features': ['open', 'high', 'low', 'close', 'volume'],
            'output_features': ['direction', 'confidence'],
            'sequence_length': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'validation_split': 0.2
        }
    }

@pytest.fixture
def mock_fetcher():
    """Create a mock data fetcher."""
    fetcher = AsyncMock()
    fetcher.fetch_latest_candle = AsyncMock(return_value={
        'timestamp': datetime.now().timestamp(),
        'open': 100.0,
        'high': 105.0,
        'low': 95.0,
        'close': 102.0,
        'volume': 1000.0
    })
    return fetcher

@pytest.fixture
def mock_router():
    """Create a mock signal router."""
    router = AsyncMock()
    router.route = AsyncMock(return_value={
        'symbol': 'ETH/USDT',
        'direction': 'long',
        'confidence': 0.8,
        'price': 100.0,
        'stop_loss': 98.0,
        'take_profit': 104.0,
        'timestamp': datetime.now(),
        'action': 'buy',
        'strategy': 'trend_following'
    })
    return router

@pytest.fixture
def mock_evaluator():
    """Create a mock signal evaluator."""
    evaluator = AsyncMock()
    evaluator.evaluate = AsyncMock(return_value=True)
    return evaluator

@pytest.fixture
def mock_executor():
    """Create a mock signal executor."""
    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=True)
    return executor

@pytest.mark.asyncio
async def test_process_multiple_symbols(mock_config, mock_fetcher, mock_router, 
                                      mock_evaluator, mock_executor):
    """Test processing multiple symbols in the trading loop."""
    # Create trading loop with mocks
    loop = TradingLoop(mock_config)
    loop.fetcher = mock_fetcher
    loop.router = mock_router
    loop.evaluator = mock_evaluator
    loop.executor = mock_executor
    
    # Process each symbol
    for symbol in mock_config['trading']['symbols']:
        # Mock the router response
        mock_router.route.return_value = {
            'symbol': symbol,
            'direction': 'long',
            'confidence': 0.8,
            'price': 100.0,
            'stop_loss': 95.0,
            'take_profit': 110.0,
            'action': 'buy',
            'strategy': 'trend_following',
            'timestamp': datetime.now()
        }
        
        # Mock the evaluator response
        mock_evaluator.evaluate.return_value = True
        
        # Mock the executor response
        mock_executor.execute.return_value = True
        
        await loop._process_symbol(symbol)
        
        # Verify fetcher was called with correct symbol
        mock_fetcher.fetch_latest_candle.assert_called_with(symbol)
        
        # Verify router was called with correct data
        mock_router.route.assert_called()
        
        # Verify evaluator was called
        mock_evaluator.evaluate.assert_called()
        
        # Verify executor was called
        mock_executor.execute.assert_called()

@pytest.mark.asyncio
async def test_portfolio_tracking_multiple_symbols():
    """Test portfolio tracking across multiple symbols."""
    portfolio = PortfolioTracker(initial_capital=10000.0)
    
    # Open positions for different symbols
    eth_position = portfolio.open_position(
        symbol='ETH/USDT',
        entry_price=2000.0,
        size=0.1,
        direction='long',
        stop_loss=1900.0,
        take_profit=2200.0
    )
    
    btc_position = portfolio.open_position(
        symbol='BTC/USDT',
        entry_price=40000.0,
        size=0.01,
        direction='short',
        stop_loss=42000.0,
        take_profit=38000.0
    )
    
    # Verify positions are tracked separately
    assert len(portfolio.get_open_positions('ETH/USDT')['ETH/USDT']) == 1
    assert len(portfolio.get_open_positions('BTC/USDT')['BTC/USDT']) == 1
    
    # Close positions
    portfolio.close_position('ETH/USDT', eth_position, 2100.0)
    portfolio.close_position('BTC/USDT', btc_position, 39000.0)
    
    # Verify PnL is tracked per symbol
    assert portfolio.get_total_pnl('ETH/USDT') > 0
    assert portfolio.get_total_pnl('BTC/USDT') > 0

@pytest.mark.asyncio
async def test_risk_management_multiple_symbols():
    """Test risk management across multiple symbols."""
    portfolio = PortfolioTracker(initial_capital=10000.0)
    risk_manager = RiskManager('config/risk_config.json', portfolio)
    
    # Test position limits
    signal1 = {
        'symbol': 'ETH/USDT',
        'direction': 'long',
        'confidence': 0.8,
        'price': 2000.0,
        'stop_loss': 1900.0,
        'take_profit': 2200.0,
        'action': 'buy',
        'strategy': 'trend_following',
        'timestamp': datetime.now()
    }
    
    signal2 = {
        'symbol': 'BTC/USDT',
        'direction': 'long',
        'confidence': 0.8,
        'price': 40000.0,
        'stop_loss': 38000.0,
        'take_profit': 42000.0,
        'action': 'buy',
        'strategy': 'trend_following',
        'timestamp': datetime.now()
    }
    
    # First signal should pass
    assert risk_manager.check(signal1)
    
    # Second signal should pass (different symbol)
    assert risk_manager.check(signal2)
    
    # Third signal for same symbol should fail
    assert not risk_manager.check(signal1)
    
    # Test drawdown limits
    portfolio.open_position(
        symbol='ETH/USDT',
        entry_price=2000.0,
        size=0.1,
        direction='long',
        stop_loss=1900.0,
        take_profit=2200.0
    )
    
    # Close position at a loss
    portfolio.close_position(
        'ETH/USDT',
        portfolio.get_open_positions('ETH/USDT')['ETH/USDT'][0],
        1800.0
    )
    
    # Next signal should fail due to drawdown
    assert not risk_manager.check(signal1)

@pytest.mark.asyncio
async def test_duplicate_signal_prevention():
    """Test prevention of duplicate signals across symbols."""
    portfolio = PortfolioTracker(initial_capital=10000.0)
    risk_manager = RiskManager('config/risk_config.json', portfolio)
    
    # Test long signals
    long_signal = {
        'symbol': 'ETH/USDT',
        'direction': 'long',
        'confidence': 0.8,
        'price': 2000.0,
        'stop_loss': 1900.0,
        'take_profit': 2200.0,
        'action': 'buy',
        'strategy': 'trend_following',
        'timestamp': datetime.now()
    }
    
    # First long signal should pass
    assert risk_manager.check(long_signal)
    
    # Immediate duplicate long should fail
    assert not risk_manager.check(long_signal)
    
    # Different symbol should pass
    long_signal['symbol'] = 'BTC/USDT'
    assert risk_manager.check(long_signal)
    
    # Test short signals
    short_signal = {
        'symbol': 'ETH/USDT',
        'direction': 'short',
        'confidence': 0.8,
        'price': 2000.0,
        'stop_loss': 2100.0,
        'take_profit': 1800.0,
        'action': 'sell',
        'strategy': 'trend_following',
        'timestamp': datetime.now()
    }
    
    # First short signal should pass (different direction)
    assert risk_manager.check(short_signal)
    
    # Immediate duplicate short should fail
    assert not risk_manager.check(short_signal)
    
    # Wait for block period
    risk_manager.last_short_signal_time['ETH/USDT'] = datetime.now() - timedelta(minutes=6)
    
    # After waiting, short signal with different price should pass
    short_signal['price'] = 2100.0
    short_signal['stop_loss'] = 2200.0
    short_signal['take_profit'] = 1900.0
    assert risk_manager.check(short_signal)

@pytest.mark.asyncio
async def test_parallel_symbol_processing():
    """Test parallel processing of symbols with different risk outcomes."""
    portfolio = PortfolioTracker(initial_capital=10000.0)
    risk_manager = RiskManager('config/risk_config.json', portfolio)
    
    # Create signals for different symbols
    eth_signal = {
        'symbol': 'ETH/USDT',
        'direction': 'long',
        'confidence': 0.8,
        'price': 2000.0,
        'stop_loss': 1900.0,
        'take_profit': 2200.0,
        'action': 'buy',
        'strategy': 'trend_following',
        'timestamp': datetime.now()
    }
    
    btc_signal = {
        'symbol': 'BTC/USDT',
        'direction': 'long',
        'confidence': 0.2,  # Low confidence
        'price': 40000.0,
        'stop_loss': 38000.0,
        'take_profit': 42000.0,
        'action': 'buy',
        'strategy': 'trend_following',
        'timestamp': datetime.now()
    }
    
    # ETH signal should pass
    assert risk_manager.check(eth_signal)
    
    # BTC signal should fail due to low confidence
    assert not risk_manager.check(btc_signal)
    
    # Verify portfolio only has ETH position
    portfolio.open_position(
        symbol='ETH/USDT',
        entry_price=eth_signal['price'],
        size=0.1,
        direction='long',
        stop_loss=eth_signal['stop_loss'],
        take_profit=eth_signal['take_profit']
    )
    
    assert len(portfolio.get_open_positions('ETH/USDT')['ETH/USDT']) == 1
    assert len(portfolio.get_open_positions('BTC/USDT')['BTC/USDT']) == 0

@pytest.mark.asyncio
async def test_max_positions_limits():
    """Test enforcement of maximum position limits both globally and per symbol."""
    portfolio = PortfolioTracker(initial_capital=10000.0)
    risk_manager = RiskManager('config/risk_config.json', portfolio)
    
    # Create signals for different symbols
    signals = [
        {
            'symbol': 'ETH/USDT',
            'direction': 'long',
            'confidence': 0.8,
            'price': 2000.0,
            'stop_loss': 1900.0,
            'take_profit': 2200.0,
            'action': 'buy',
            'strategy': 'trend_following',
            'timestamp': datetime.now()
        },
        {
            'symbol': 'BTC/USDT',
            'direction': 'long',
            'confidence': 0.8,
            'price': 40000.0,
            'stop_loss': 38000.0,
            'take_profit': 42000.0,
            'action': 'buy',
            'strategy': 'trend_following',
            'timestamp': datetime.now()
        },
        {
            'symbol': 'SOL/USDT',
            'direction': 'long',
            'confidence': 0.8,
            'price': 100.0,
            'stop_loss': 95.0,
            'take_profit': 110.0,
            'action': 'buy',
            'strategy': 'trend_following',
            'timestamp': datetime.now()
        },
        {
            'symbol': 'ETH/USDT',  # Duplicate symbol
            'direction': 'long',
            'confidence': 0.8,
            'price': 2000.0,
            'stop_loss': 1900.0,
            'take_profit': 2200.0,
            'action': 'buy',
            'strategy': 'trend_following',
            'timestamp': datetime.now()
        }
    ]
    
    # First three signals should pass (different symbols)
    for signal in signals[:3]:
        assert risk_manager.check(signal)
        portfolio.open_position(
            symbol=signal['symbol'],
            entry_price=signal['price'],
            size=0.1,
            direction=signal['direction'],
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit']
        )
    
    # Fourth signal should fail (max positions per symbol)
    assert not risk_manager.check(signals[3])
    
    # Verify position counts
    assert len(portfolio.get_open_positions('ETH/USDT')['ETH/USDT']) == 1
    assert len(portfolio.get_open_positions('BTC/USDT')['BTC/USDT']) == 1
    assert len(portfolio.get_open_positions('SOL/USDT')['SOL/USDT']) == 1
    
    # Try to open another position (should fail due to global limit)
    new_signal = {
        'symbol': 'DOGE/USDT',
        'direction': 'long',
        'confidence': 0.8,
        'price': 0.1,
        'stop_loss': 0.095,
        'take_profit': 0.11,
        'action': 'buy',
        'strategy': 'trend_following',
        'timestamp': datetime.now()
    }
    assert not risk_manager.check(new_signal) 