import pytest
import torch
import pandas as pd
import numpy as np
from ml.train.dataset_builder import TimeSeriesDataset

@pytest.fixture
def sample_data(tmp_path):
    """Create sample data for testing."""
    # Create sample features
    features = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    
    # Create sample labels
    labels = pd.DataFrame({
        'label': np.random.choice([0, 1, 2], size=100)
    })
    
    # Save to temporary files
    features_path = tmp_path / "test_features.csv"
    labels_path = tmp_path / "test_labels.csv"
    
    features.to_csv(features_path, index=False)
    labels.to_csv(labels_path, index=False)
    
    return str(features_path), str(labels_path)

def test_dataset_initialization(sample_data):
    """Test dataset initialization."""
    features_path, labels_path = sample_data
    seq_len = 60
    
    ds = TimeSeriesDataset(
        features_path=features_path,
        labels_path=labels_path,
        seq_len=seq_len
    )
    
    assert len(ds) == 40  # 100 - 60 = 40 samples
    assert ds.seq_len == seq_len

def test_dataset_getitem(sample_data):
    """Test getting items from the dataset."""
    features_path, labels_path = sample_data
    ds = TimeSeriesDataset(features_path, labels_path, seq_len=60)
    
    x, y = ds[0]
    
    # Check shapes and types
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (60, 3)  # (seq_len, num_features)
    assert x.dtype == torch.float32
    assert y.dtype == torch.long
    assert y.item() in [0, 1, 2]

def test_dataset_feature_names(sample_data):
    """Test getting feature names."""
    features_path, labels_path = sample_data
    ds = TimeSeriesDataset(features_path, labels_path)
    
    feature_names = ds.get_feature_names()
    assert len(feature_names) == 3
    assert all(name in feature_names for name in ['feature1', 'feature2', 'feature3'])

def test_dataset_label_distribution(sample_data):
    """Test getting label distribution."""
    features_path, labels_path = sample_data
    ds = TimeSeriesDataset(features_path, labels_path)
    
    label_dist = ds.get_label_distribution()
    assert isinstance(label_dist, dict)
    assert all(label in [0, 1, 2] for label in label_dist.keys())
    assert sum(label_dist.values()) == 100  # Total number of labels

def test_dataset_empty_files(tmp_path):
    """Test handling of empty files."""
    empty_features = tmp_path / "empty_features.csv"
    empty_labels = tmp_path / "empty_labels.csv"
    
    pd.DataFrame().to_csv(empty_features, index=False)
    pd.DataFrame().to_csv(empty_labels, index=False)
    
    with pytest.raises(ValueError):
        TimeSeriesDataset(str(empty_features), str(empty_labels))

def test_dataset_invalid_sequence_length(sample_data):
    """Test handling of invalid sequence length."""
    features_path, labels_path = sample_data
    
    # Test negative sequence length
    with pytest.raises(ValueError):
        TimeSeriesDataset(features_path, labels_path, seq_len=-1)
    
    # Test zero sequence length
    with pytest.raises(ValueError):
        TimeSeriesDataset(features_path, labels_path, seq_len=0)
    
    # Test sequence length equal to data length
    with pytest.raises(ValueError):
        TimeSeriesDataset(features_path, labels_path, seq_len=100)
    
    # Test sequence length greater than data length
    with pytest.raises(ValueError):
        TimeSeriesDataset(features_path, labels_path, seq_len=101)

def test_dataset_index_out_of_range(sample_data):
    """Test handling of out-of-range indices."""
    features_path, labels_path = sample_data
    ds = TimeSeriesDataset(features_path, labels_path, seq_len=60)
    
    # Test negative index
    with pytest.raises(IndexError):
        _ = ds[-1]
    
    # Test index equal to length
    with pytest.raises(IndexError):
        _ = ds[len(ds)]
    
    # Test index greater than length
    with pytest.raises(IndexError):
        _ = ds[len(ds) + 1] 