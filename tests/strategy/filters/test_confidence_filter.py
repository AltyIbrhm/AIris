import pytest
import numpy as np
from strategy.filters.confidence_filter import ConfidenceFilter

def test_confidence_initialization():
    """Test confidence filter initialization with default and custom parameters"""
    # Test default parameters
    cf = ConfidenceFilter()
    assert cf.threshold == 0.85
    
    # Test custom parameters
    cf = ConfidenceFilter(threshold=0.90)
    assert cf.threshold == 0.90

def test_confidence_check():
    """Test confidence check with various confidence values"""
    cf = ConfidenceFilter(threshold=0.85)
    
    # Test above threshold
    assert cf.check_entry(0.90, "BUY") == True
    assert cf.check_entry(0.95, "SELL") == True
    assert cf.check_entry(1.0, "BUY") == True
    
    # Test below threshold
    assert cf.check_entry(0.80, "BUY") == False
    assert cf.check_entry(0.84, "SELL") == False
    assert cf.check_entry(0.0, "BUY") == False

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    cf = ConfidenceFilter(threshold=0.85)
    
    # Test exactly at threshold
    assert cf.check_entry(0.85, "BUY") == False
    
    # Test invalid confidence values
    assert cf.check_entry(-0.1, "BUY") == False
    assert cf.check_entry(1.1, "SELL") == False
    
    # Test invalid signals
    assert cf.check_entry(0.90, "HOLD") == False
    assert cf.check_entry(0.90, "") == False

def test_realistic_confidence_series():
    """Test confidence filter with realistic confidence values"""
    cf = ConfidenceFilter(threshold=0.85)
    
    # Simulate a series of confidence values with noise
    np.random.seed(42)  # For reproducibility
    base_confidence = 0.90
    noise_level = 0.05
    
    confidences = []
    for _ in range(20):
        noise = np.random.normal(0, noise_level)
        confidence = min(max(base_confidence + noise, 0), 1)  # Clamp between 0 and 1
        confidences.append(confidence)
    
    # Test each confidence value
    for confidence in confidences:
        result = cf.check_entry(confidence, "BUY")
        if confidence > 0.85:
            assert result == True
        else:
            assert result == False
    
    # Test string representation
    assert str(cf) == "ConfidenceFilter(threshold=0.85)" 