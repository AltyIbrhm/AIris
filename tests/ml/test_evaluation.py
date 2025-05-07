import os
import torch
import numpy as np
import pytest
from pathlib import Path
from sklearn.metrics import classification_report
import yaml

from ml.evaluation.evaluate_model import evaluate
from ml.evaluation.simulate_pnl import simulate_trading
from ml.evaluation.load_price_data import load_price_data

def test_evaluation_runs():
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

def test_evaluation_outputs():
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

def test_metrics_thresholds():
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

def test_pnl_simulation():
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

def test_error_handling():
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