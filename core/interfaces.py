"""
Defines base interfaces for Strategy, RiskManager, MarketDataFetcher, and other core components.
These interfaces ensure modularity and testability.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class Strategy(ABC):
    """Base interface for all trading strategies."""
    
    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on market data."""
        pass

class Model(ABC):
    """Base interface for ML models."""
    
    @abstractmethod
    def predict(self, data: Dict[str, Any]) -> float:
        """Make predictions based on input data."""
        pass
    
    @abstractmethod
    def train(self, data: Dict[str, Any]) -> None:
        """Train the model on input data."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk."""
        pass

class Exchange(ABC):
    """Base interface for cryptocurrency exchanges."""
    
    @abstractmethod
    def get_market_data(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get market data for a symbol and timeframe."""
        pass
    
    @abstractmethod
    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Place a trading order."""
        pass
    
    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        pass

class RiskManager(ABC):
    """Base interface for risk management."""
    
    @abstractmethod
    async def filter_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter trading signals based on risk parameters."""
        pass
    
    @abstractmethod
    async def evaluate_risk(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate risk for a given position."""
        pass

class MarketDataFetcher(ABC):
    """Base interface for market data retrieval."""
    
    @abstractmethod
    async def fetch_latest(self) -> Dict[str, Any]:
        """Fetch the latest market data."""
        pass
    
    @abstractmethod
    async def fetch_data(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Fetch market data for a given symbol and interval."""
        pass

class SignalRouter(ABC):
    """Base interface for routing signals from strategies."""
    
    @abstractmethod
    async def get_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get trading signals from all active strategies."""
        pass
    
    @abstractmethod
    def add_strategy(self, strategy: Strategy) -> None:
        """Add a strategy to the router."""
        pass
    
    @abstractmethod
    def remove_strategy(self, strategy_name: str) -> None:
        """Remove a strategy from the router."""
        pass

class PaperTradingEngine(ABC):
    """Base interface for paper trading execution."""
    
    @abstractmethod
    async def execute_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute trading signals in paper trading mode."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current paper trading positions."""
        pass
    
    @abstractmethod
    async def get_pnl(self) -> Dict[str, float]:
        """Get current paper trading PnL."""
        pass

class OrderExecutor(ABC):
    """Base interface for order execution."""
    
    @abstractmethod
    def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trading order."""
        pass 