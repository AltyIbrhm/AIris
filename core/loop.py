"""
Core trading loop implementation.
"""
import asyncio
import logging
from typing import Optional

from config.config_schema import Config
from core.interfaces import MarketDataFetcher, SignalRouter, RiskManager, PaperTradingEngine

logger = logging.getLogger("airis")

class TradingLoop:
    """Main trading loop that orchestrates all components."""
    
    def __init__(self, config: Config):
        """Initialize the trading loop with configuration."""
        self.config = config
        self.market_data: Optional[MarketDataFetcher] = None
        self.signal_router: Optional[SignalRouter] = None
        self.risk_manager: Optional[RiskManager] = None
        self.trading_engine: Optional[PaperTradingEngine] = None
        self.is_running = False
        
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing trading components...")
        
        # Initialize components (to be implemented)
        # self.market_data = MarketDataFetcher(self.config)
        # self.signal_router = SignalRouter(self.config)
        # self.risk_manager = RiskManager(self.config)
        # self.trading_engine = PaperTradingEngine(self.config)
        
        logger.info("Trading components initialized")
    
    async def run(self):
        """Run the main trading loop."""
        if not all([self.market_data, self.signal_router, 
                   self.risk_manager, self.trading_engine]):
            raise RuntimeError("Trading components not initialized")
        
        self.is_running = True
        logger.info("Starting trading loop")
        
        try:
            while self.is_running:
                # 1. Fetch latest market data
                market_data = await self.market_data.fetch_latest()
                
                # 2. Get signals from strategies
                signals = await self.signal_router.get_signals(market_data)
                
                # 3. Apply risk filters
                filtered_signals = await self.risk_manager.filter_signals(signals)
                
                # 4. Execute trades
                if filtered_signals:
                    await self.trading_engine.execute_signals(filtered_signals)
                
                # 5. Log results
                self._log_iteration(market_data, signals, filtered_signals)
                
                # Wait for next iteration
                await asyncio.sleep(self.config.trading.interval)
                
        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}", exc_info=True)
            raise
        finally:
            self.is_running = False
            logger.info("Trading loop stopped")
    
    def stop(self):
        """Stop the trading loop."""
        self.is_running = False
        logger.info("Stopping trading loop...")
    
    def _log_iteration(self, market_data, signals, filtered_signals):
        """Log the results of a trading loop iteration."""
        logger.debug(f"Market data: {market_data}")
        logger.debug(f"Generated signals: {len(signals)}")
        logger.debug(f"Filtered signals: {len(filtered_signals)}")

async def run_trading_loop(config: Config):
    """
    Run the main trading loop.
    
    Args:
        config: System configuration
    """
    loop = TradingLoop(config)
    await loop.initialize()
    await loop.run() 