"""
Entrypoint for the AIris bot.
Initializes configuration, components, and starts the main engine.
"""
from typing import Dict, Any
from .engine import TradingEngine

class BotRunner:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the bot runner with configuration."""
        self.config = config
        self.engine = TradingEngine()

    def start(self):
        """Start the trading bot."""
        try:
            self.engine.run()
        except KeyboardInterrupt:
            self.engine.stop()
        except Exception as e:
            print(f"Error running bot: {e}")
            self.engine.stop()

if __name__ == "__main__":
    # Example usage
    config = {
        "api_key": "your_api_key",
        "api_secret": "your_api_secret",
        "trading_pair": "BTCUSDT"
    }
    runner = BotRunner(config)
    runner.start() 