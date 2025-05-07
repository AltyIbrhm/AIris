import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_price_data(features_path: str) -> np.ndarray:
    """
    Load price data from features file.
    
    Args:
        features_path: Path to the features CSV file
        
    Returns:
        Array of closing prices
    """
    try:
        # Load features data
        df = pd.read_csv(features_path)
        
        # Extract closing price column
        if 'close' in df.columns:
            prices = df['close'].values
        else:
            raise ValueError("No 'close' column found in features data")
        
        logger.info(f"Loaded {len(prices)} price points")
        return prices
        
    except Exception as e:
        logger.error(f"Error loading price data: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    from ml.evaluation.evaluate_model import evaluate
    from ml.evaluation.simulate_pnl import simulate_trading
    
    # Load configuration
    import yaml
    with open("config/model_config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Get model predictions
    eval_results = evaluate()
    
    # Load price data
    prices = load_price_data(config["features_path"])
    
    # Run simulation
    metrics = simulate_trading(
        eval_results["predictions"],
        eval_results["probabilities"],
        prices
    ) 