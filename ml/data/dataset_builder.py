import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data with features and labels.
    """
    def __init__(self, features_path: str, labels_path: str, seq_len: int = 60):
        """
        Initialize the dataset.
        
        Args:
            features_path: Path to features CSV
            labels_path: Path to labels CSV
            seq_len: Length of sequence for each sample
        """
        # Load data
        self.features = pd.read_csv(features_path)
        self.labels = pd.read_csv(labels_path)
        self.seq_len = seq_len
        
        # Store timestamps
        self.timestamps = pd.to_datetime(self.features['timestamp'])
        
        # Align data using timestamps
        self.features['timestamp'] = pd.to_datetime(self.features['timestamp'])
        self.labels['timestamp'] = pd.to_datetime(self.labels['timestamp'])
        
        # Merge on timestamp and ensure alignment
        merged = pd.merge(self.features, self.labels[['timestamp', 'label']], on='timestamp', how='inner')
        self.features = merged[self.features.columns]
        self.labels = merged[['timestamp', 'label']]
        
        logger.info(f"Aligned data using timestamps. New lengths - Features: {len(self.features)}, Labels: {len(self.labels)}")
        
        # Map labels to 0, 1, 2
        label_map = {-1: 0, 0: 1, 1: 2}  # SELL: 0, HOLD: 1, BUY: 2
        self.labels['label'] = self.labels['label'].map(label_map)
        
        # Drop non-numeric columns
        numeric_cols = self.features.select_dtypes(include=[np.number]).columns
        non_numeric_cols = set(self.features.columns) - set(numeric_cols)
        if non_numeric_cols:
            logger.warning(f"Dropping non-numeric columns: {non_numeric_cols}")
            self.features = self.features[numeric_cols]
        
        # Convert to numpy arrays
        self.features = self.features.values
        self.labels = self.labels['label'].values
        
        logger.info(f"Dataset initialized with {len(self) } sequences")
        logger.info(f"Feature shape: {self.features.shape}")
        
        # Log label distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        label_dist = dict(zip(unique, counts))
        logger.info(f"Label distribution: {label_dist}")
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return max(0, len(self.features) - self.seq_len)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sequence and its label.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (sequence, label)
        """
        # Get sequence
        sequence = self.features[idx:idx + self.seq_len]
        
        # Get label (use the label at the end of the sequence)
        label = self.labels[idx + self.seq_len - 1]
        
        return torch.FloatTensor(sequence), label
    
    def get_feature_names(self) -> list:
        """Return the names of the features."""
        return self.features.columns.tolist()
    
    def get_label_distribution(self) -> dict:
        """Return the distribution of labels."""
        return self.labels.value_counts().to_dict() 