"""
AIris Trading System - Main Entry Point
"""
import asyncio
import logging
from typing import Optional

from config.config_schema import load_config
from core.loop import run_trading_loop
from utils.logger import setup_logger

async def main():
    """Main entry point for the trading system."""
    # Load configuration
    config = load_config()
    
    # Setup logging
    logger = setup_logger("airis", config.logging.log_dir)
    logger.info("Starting AIris Trading System")
    
    try:
        # Run the main trading loop
        await run_trading_loop(config)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        logger.info("Shutting down AIris Trading System")

if __name__ == "__main__":
    asyncio.run(main()) 