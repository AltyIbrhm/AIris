import pandas as pd
import torch
from torch.utils.data import Dataset
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    def __init__(self, features_path, labels_path, seq_len=60):
        """
        Initialize the TimeSeriesDataset.
        
        Args:
            features_path (str): Path to the features CSV file
            labels_path (str): Path to the labels CSV file
            seq_len (int): Length of the sequence window
        """
        try:
            # Load data
            self.features = pd.read_csv(features_path)
            self.labels = pd.read_csv(labels_path)
            
            # Validate sequence length
            if seq_len <= 0:
                raise ValueError("Sequence length must be positive")
            
            # Validate data
            if len(self.features) == 0 or len(self.labels) == 0:
                raise ValueError("Empty features or labels file")

            # Align to the shortest length
            min_len = min(len(self.features), len(self.labels))
            self.features = self.features.iloc[:min_len].reset_index(drop=True)
            self.labels = self.labels.iloc[:min_len].reset_index(drop=True)
            
            # Validate sequence length against data length
            if seq_len >= min_len:
                raise ValueError(f"Sequence length ({seq_len}) must be less than data length ({min_len})")
            
            self.seq_len = seq_len

            # Log dataset info
            logger.info(f"Dataset initialized with {len(self)} samples")
            logger.info(f"Feature shape: {self.features.shape}")
            logger.info(f"Label distribution: {self.labels['label'].value_counts().to_dict()}")

        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            raise

    def __len__(self):
        """Return the number of samples in the dataset."""
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (x_seq, y_label) where:
                x_seq: tensor of shape (seq_len, num_features)
                y_label: tensor of shape (1,) containing the class label
        """
        try:
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range [0, {len(self)})")
                
            # Get sequence of features
            x_seq = self.features.iloc[idx : idx + self.seq_len].values
            
            # Get corresponding label
            y_label = self.labels.iloc[idx + self.seq_len]["label"]
            
            # Convert to tensors
            x_tensor = torch.tensor(x_seq, dtype=torch.float32)
            y_tensor = torch.tensor(y_label, dtype=torch.long)
            
            return x_tensor, y_tensor
            
        except Exception as e:
            logger.error(f"Error getting item at index {idx}: {str(e)}")
            raise

    def get_feature_names(self):
        """Return the names of the features."""
        return self.features.columns.tolist()

    def get_label_distribution(self):
        """Return the distribution of labels in the dataset."""
        return self.labels['label'].value_counts().to_dict() 