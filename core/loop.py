"""
Trading loop implementation.
"""
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
from config.schema import load_config
from data.fetcher import DataFetcher
from core.router import SignalRouter
from core.evaluator import SignalEvaluator
from core.executor import SignalExecutor
from risk.checker import RiskManager
from utils.portfolio import PortfolioTracker
from config.schema import Config

logger = logging.getLogger(__name__)

class TradingLoop:
    """Main trading loop implementation."""
    
    def __init__(self, config: str | Dict[str, Any]):
        """Initialize trading loop with configuration."""
        if isinstance(config, str):
            self.config = load_config(config)
        else:
            self.config = Config.parse_obj(config).trading
        self.fetcher = DataFetcher(self.config.exchange)
        self.router = SignalRouter()
        self.evaluator = SignalEvaluator()
        self.portfolio = PortfolioTracker()
        self.risk_manager = RiskManager('config/risk_config.json', self.portfolio)
        self.executor = SignalExecutor(self.config.paper_trading)
        
    async def run(self):
        """Run the trading loop."""
        logger.info("Starting trading loop...")
        
        while True:
            try:
                # Process each symbol independently
                for symbol in self.config.symbols:
                    await self._process_symbol(symbol)
                    
                # Wait for next iteration
                await asyncio.sleep(self.config.poll_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
                
    async def _process_symbol(self, symbol: str):
        """Process a single symbol."""
        try:
            # Fetch latest candle
            candle = await self.fetcher.fetch_latest_candle(symbol)
            if not candle:
                logger.warning(f"No candle data for {symbol}")
                return
                
            # Route signal
            signal = await self.router.route(symbol, candle)
            if not signal:
                return
                
            # Evaluate signal
            if not await self.evaluator.evaluate(signal):
                return
                
            # Check risk parameters
            if not self.risk_manager.check(signal):
                return
                
            # Calculate position size
            position_size = self.risk_manager.get_position_size(signal)
            
            # Execute trade
            await self.executor.execute(signal, position_size)
            
            # Update portfolio
            if signal['direction'] == 'long':
                self.portfolio.open_position(
                    symbol=symbol,
                    entry_price=signal['price'],
                    size=position_size,
                    direction='long',
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit']
                )
            else:  # short
                self.portfolio.open_position(
                    symbol=symbol,
                    entry_price=signal['price'],
                    size=position_size,
                    direction='short',
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit']
                )
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            
    async def stop(self):
        """Stop the trading loop."""
        logger.info("Stopping trading loop...")
        await self.fetcher.close()
        await self.executor.close()
        
def run_trading_loop(config_path: str):
    """Run the trading loop with given configuration."""
    loop = TradingLoop(config_path)
    try:
        asyncio.run(loop.run())
    except KeyboardInterrupt:
        logger.info("Received stop signal")
    finally:
        asyncio.run(loop.stop()) 