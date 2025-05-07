import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data with features and labels.
    """
    def __init__(self, features_path: str, labels_path: str, seq_len: int = 60):
        """
        Initialize the dataset.
        
        Args:
            features_path (str): Path to the features CSV file
            labels_path (str): Path to the labels CSV file
            seq_len (int): Length of the sequence window
        """
        # Load features and labels
        self.features = pd.read_csv(features_path)
        self.labels = pd.read_csv(labels_path)
        
        # Align timestamps if they exist
        if 'timestamp' in self.features.columns and 'timestamp' in self.labels.columns:
            # Convert timestamps to datetime
            self.features['timestamp'] = pd.to_datetime(self.features['timestamp'])
            self.labels['timestamp'] = pd.to_datetime(self.labels['timestamp'])
            
            # Merge on timestamp
            merged_data = pd.merge(
                self.features,
                self.labels[['timestamp', 'label']],
                on='timestamp',
                how='inner'
            )
            
            # Split back into features and labels
            self.features = merged_data.drop('label', axis=1)
            self.labels = merged_data[['timestamp', 'label']]
            
            logger.info(f"Aligned data using timestamps. New lengths - Features: {len(self.features)}, Labels: {len(self.labels)}")
        
        # Validate data
        if len(self.features) != len(self.labels):
            raise ValueError(f"Features length ({len(self.features)}) does not match labels length ({len(self.labels)})")
        
        # Convert labels to numeric if they're not already
        if not pd.api.types.is_numeric_dtype(self.labels['label']):
            self.labels['label'] = pd.Categorical(self.labels['label']).codes
        
        # Map labels to 0, 1, 2 (SELL, HOLD, BUY)
        label_map = {-1: 0, 0: 1, 1: 2}  # Map -1 to 0 (SELL), 0 to 1 (HOLD), 1 to 2 (BUY)
        self.labels['label'] = self.labels['label'].map(label_map)
        
        # Handle non-numeric columns in features
        numeric_columns = self.features.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < len(self.features.columns):
            logger.warning(f"Dropping non-numeric columns: {set(self.features.columns) - set(numeric_columns)}")
            self.features = self.features[numeric_columns]
        
        # Store sequence length
        self.seq_len = seq_len
        
        # Calculate number of sequences
        self.n_sequences = len(self.features) - seq_len + 1
        
        logger.info(f"Dataset initialized with {self.n_sequences} sequences")
        logger.info(f"Feature shape: {self.features.shape}")
        logger.info(f"Label distribution: {self.labels['label'].value_counts().to_dict()}")
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return self.n_sequences
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a sequence of features and its corresponding label.
        
        Args:
            idx (int): Index of the sequence
            
        Returns:
            tuple: (features_tensor, label)
        """
        if idx < 0 or idx >= self.n_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self.n_sequences})")
        
        # Get sequence of features
        feature_seq = self.features.iloc[idx:idx + self.seq_len]
        
        # Get the label for the last timestep in the sequence
        label = self.labels.iloc[idx + self.seq_len - 1]['label']
        
        # Convert features to tensor
        feature_tensor = torch.FloatTensor(feature_seq.values)
        
        return feature_tensor, label
    
    def get_feature_names(self) -> list:
        """Return the names of the features."""
        return self.features.columns.tolist()
    
    def get_label_distribution(self) -> dict:
        """Return the distribution of labels."""
        return self.labels['label'].value_counts().to_dict() 