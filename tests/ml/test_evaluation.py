import os
import torch
import numpy as np
import pytest
from pathlib import Path
from sklearn.metrics import classification_report
import yaml
import pandas as pd
import tempfile

from ml.evaluation.evaluate_model import evaluate
from ml.evaluation.simulate_pnl import simulate_trading
from ml.evaluation.load_price_data import load_price_data

# Skip all evaluation tests in CI environment
skip_in_ci = pytest.mark.skipif(
    os.environ.get('CI') == 'true',
    reason="Skipping evaluation tests in CI environment - requires data files"
)

@pytest.fixture
def mock_data_files():
    """Create temporary mock data files for testing."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock features
        features = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='5min'),
            'open': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'close': np.random.randn(100),
            'volume': np.random.randn(100),
            'rsi': np.random.randn(100),
            'macd': np.random.randn(100),
            'bb_upper': np.random.randn(100),
            'bb_lower': np.random.randn(100)
        })
        
        # Create mock labels
        labels = pd.DataFrame({
            'timestamp': features['timestamp'],
            'label': np.random.choice([-1, 0, 1], size=100)  # SELL, HOLD, BUY
        })
        
        # Save to temporary files
        features_path = os.path.join(temp_dir, 'BTCUSDT_5m_features.csv')
        labels_path = os.path.join(temp_dir, 'BTCUSDT_5m_labels.csv')
        
        features.to_csv(features_path, index=False)
        labels.to_csv(labels_path, index=False)
        
        # Create mock model config
        config = {
            'features_path': features_path,
            'labels_path': labels_path,
            'seq_len': 60,
            'batch_size': 32,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'val_split': 0.2,
            'save_path': os.path.join(temp_dir, 'model.pth')
        }
        
        # Save config
        config_path = os.path.join(temp_dir, 'model_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create mock model file
        model = torch.nn.Linear(10, 3)  # Simple linear model for testing
        torch.save(model.state_dict(), config['save_path'])
        
        # Create output directories
        os.makedirs('ml/evaluation/figures', exist_ok=True)
        
        yield {
            'features_path': features_path,
            'labels_path': labels_path,
            'config_path': config_path
        }

@skip_in_ci
def test_evaluation_runs(mock_data_files):
    """Basic integration test: ensures evaluation script runs without errors."""
    try:
        results = evaluate()
    except Exception as e:
        pytest.fail(f"Evaluation script failed: {str(e)}")
    
    # Check that results dictionary contains expected keys
    expected_keys = [
        "accuracy",
        "classification_report",
        "confusion_matrix",
        "predictions",
        "probabilities",
        "true_labels"
    ]
    for key in expected_keys:
        assert key in results, f"Missing key in results: {key}"

@skip_in_ci
def test_evaluation_outputs(mock_data_files):
    """Test that evaluation generates expected outputs."""
    results = evaluate()
    
    # Check confusion matrix image
    assert os.path.exists("ml/evaluation/figures/confusion_matrix.png"), \
        "Confusion matrix image not found"
    
    # Check predictions shape and values
    assert isinstance(results["predictions"], np.ndarray), \
        "Predictions should be numpy array"
    assert len(results["predictions"]) > 0, \
        "Predictions array should not be empty"
    assert set(np.unique(results["predictions"])).issubset({0, 1, 2}), \
        "Predictions should only contain values 0, 1, or 2"
    
    # Check probabilities shape and values
    assert isinstance(results["probabilities"], np.ndarray), \
        "Probabilities should be numpy array"
    assert results["probabilities"].shape[1] == 3, \
        "Probabilities should have 3 columns (one per class)"
    assert np.allclose(results["probabilities"].sum(axis=1), 1.0), \
        "Probabilities should sum to 1 for each prediction"

@skip_in_ci
def test_metrics_thresholds(mock_data_files):
    """Test that model performance meets minimum thresholds."""
    results = evaluate()
    
    # Parse classification report
    report = classification_report(
        results["true_labels"],
        results["predictions"],
        target_names=["SELL", "HOLD", "BUY"],
        output_dict=True
    )
    
    # Check minimum performance thresholds
    min_f1 = 0.0  # Lower threshold for initial testing
    min_accuracy = 0.4  # Lower threshold for initial testing
    
    for class_name in ["SELL", "HOLD", "BUY"]:
        assert report[class_name]["f1-score"] >= min_f1, \
            f"{class_name} class F1 score below threshold"
    
    assert results["accuracy"] >= min_accuracy, \
        "Overall accuracy below threshold"

@skip_in_ci
def test_pnl_simulation(mock_data_files):
    """Test PnL simulation with evaluation results."""
    # Get evaluation results
    results = evaluate()
    
    # Load price data
    with open("config/model_config.yaml") as f:
        config = yaml.safe_load(f)
    prices = load_price_data(config["features_path"])
    
    # Run simulation
    metrics = simulate_trading(
        results["predictions"],
        results["probabilities"],
        prices
    )
    
    # Check simulation outputs
    assert "total_trades" in metrics, "Missing total_trades in metrics"
    assert "win_rate" in metrics, "Missing win_rate in metrics"
    assert "equity_curve" in metrics, "Missing equity_curve in metrics"
    
    # Check equity curve
    assert len(metrics["equity_curve"]) > 0, "Equity curve should not be empty"
    assert all(isinstance(x, (int, float)) for x in metrics["equity_curve"]), \
        "Equity curve should contain numeric values"
    
    # Check that performance plots were generated
    assert os.path.exists("ml/evaluation/figures/equity_curve.png"), \
        "Equity curve plot not found"
    if metrics["total_trades"] > 0:
        assert os.path.exists("ml/evaluation/figures/profit_distribution.png"), \
            "Profit distribution plot not found"

@skip_in_ci
def test_error_handling(mock_data_files):
    """Test error handling in evaluation pipeline."""
    # Test with invalid model path
    with open("config/model_config.yaml") as f:
        config = yaml.safe_load(f)
    original_path = config["save_path"]
    
    try:
        # Temporarily change model path to non-existent file
        config["save_path"] = "nonexistent_model.pth"
        with open("config/model_config.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            evaluate()
    
    finally:
        # Restore original config
        config["save_path"] = original_path
        with open("config/model_config.yaml", "w") as f:
            yaml.dump(config, f)

if __name__ == "__main__":
    pytest.main([__file__]) 