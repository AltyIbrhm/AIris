from ml.train.dataset_builder import TimeSeriesDataset
from torch.utils.data import DataLoader
import torch

def test_with_real_data():
    try:
        # Initialize dataset
        dataset = TimeSeriesDataset(
            features_path="ml/data/processed/BTCUSDT_5m_features.csv",
            labels_path="ml/data/processed/BTCUSDT_5m_labels.csv",
            seq_len=60
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0  # Set to 0 for debugging
        )
        
        # Test batch iteration
        for batch_idx, (x, y) in enumerate(dataloader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"Input shape: {x.shape}")  # Should be (batch_size, seq_len, num_features)
            print(f"Label shape: {y.shape}")  # Should be (batch_size,)
            
            # Count labels in the batch
            labels, counts = torch.unique(y, return_counts=True)
            label_dist = {int(l.item()): int(c.item()) for l, c in zip(labels, counts)}
            print(f"Label distribution in batch: {label_dist}")
            
            if batch_idx == 0:  # Print details for first batch only
                print("\nDetailed first batch info:")
                print(f"Input dtype: {x.dtype}")
                print(f"Label dtype: {y.dtype}")
                print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")
                print(f"Unique labels: {torch.unique(y).tolist()}")
            
            if batch_idx >= 2:  # Only show first 3 batches
                break
                
    except FileNotFoundError as e:
        print(f"Error: Data files not found. Please ensure the processed data exists.\n{str(e)}")
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    test_with_real_data() 