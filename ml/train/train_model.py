import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from ml.data.dataset_builder import TimeSeriesDataset
from ml.models.lstm_model import LSTMClassifier
import yaml
import os
import logging
from typing import Dict, Any
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(path: str = "config/model_config.yaml") -> Dict[str, Any]:
    """
    Load model configuration from YAML file.
    
    Args:
        path (str): Path to the config file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config from {path}: {str(e)}")
        raise

def print_dataset_stats(dataset: TimeSeriesDataset) -> None:
    """
    Print dataset statistics including class distribution.
    
    Args:
        dataset (TimeSeriesDataset): The dataset to analyze
    """
    # Get label distribution
    labels = dataset.labels["label"].value_counts()
    total_samples = len(dataset)
    
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"Total samples: {total_samples}")
    logger.info("\nLabel Distribution:")
    for label, count in labels.items():
        percentage = (count / total_samples) * 100
        logger.info(f"Class {label}: {count} samples ({percentage:.2f}%)")
    logger.info("========================\n")

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various classification metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    metrics = {}
    
    # Calculate F1 score
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Calculate per-class accuracy
    for i in range(len(cm)):
        metrics[f'class_{i}_accuracy'] = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
    
    return metrics

def train_model() -> None:
    """
    Main training function that handles:
    - Dataset loading and splitting
    - Model initialization
    - Training loop with validation
    - Model checkpointing
    """
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded successfully")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Initialize dataset
        dataset = TimeSeriesDataset(
            features_path=config["features_path"],
            labels_path=config["labels_path"],
            seq_len=config["seq_len"]
        )
        input_dim = dataset[0][0].shape[1]
        logger.info(f"Dataset initialized with {len(dataset)} samples")
        logger.info(f"Input dimension: {input_dim}")
        
        # Print dataset statistics
        print_dataset_stats(dataset)

        # Split dataset
        val_size = int(len(dataset) * config["val_split"])
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        logger.info(f"Train size: {train_size}, Validation size: {val_size}")

        # Create dataloaders
        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0  # Adjust based on system
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0  # Adjust based on system
        )

        # Initialize model
        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        ).to(device)
        logger.info("Model initialized")

        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        class_weights = torch.tensor(config["class_weights"], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Create checkpoint directory
        os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
        
        # Training loop
        best_val_loss = float("inf")
        early_stopping_patience = 5
        early_stopping_counter = 0
        
        logger.info("Starting training...")
        for epoch in range(config["epochs"]):
            # Training phase
            model.train()
            total_loss = 0
            correct_train = 0
            total_train = 0
            train_preds = []
            train_true = []
            
            for batch_idx, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                total_train += y.size(0)
                correct_train += (predicted == y).sum().item()
                
                # Store predictions for metrics
                train_preds.extend(predicted.cpu().numpy())
                train_true.extend(y.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    logger.debug(f"Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}")

            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = 100 * correct_train / total_train
            train_metrics = calculate_metrics(np.array(train_true), np.array(train_preds))

            # Validation phase
            model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    val_logits = model(X)
                    val_loss += criterion(val_logits, y).item()
                    
                    _, predicted = torch.max(val_logits.data, 1)
                    total_val += y.size(0)
                    correct_val += (predicted == y).sum().item()
                    
                    # Store predictions for metrics
                    val_preds.extend(predicted.cpu().numpy())
                    val_true.extend(y.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct_val / total_val
            val_metrics = calculate_metrics(np.array(val_true), np.array(val_preds))

            # Logging
            logger.info(
                f"[Epoch {epoch+1}/{config['epochs']}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.2f}% | "
                f"Train F1: {train_metrics['f1']:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.2f}% | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Log per-class validation accuracy
            logger.info("Validation Class Accuracies:")
            for i in range(3):  # Assuming 3 classes (SELL, HOLD, BUY)
                logger.info(f"Class {i}: {val_metrics[f'class_{i}_accuracy']:.2%}")

            # Model checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_metrics': val_metrics,
                    'config': config
                }, config["save_path"])
                
                logger.info(f"✅ New best model saved: {config['save_path']}")
            else:
                early_stopping_counter += 1
                
            # Early stopping check
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        logger.info("✅ Training complete.")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # Verify model file exists
        if os.path.exists(config["save_path"]):
            file_size = os.path.getsize(config["save_path"]) / (1024 * 1024)  # Convert to MB
            logger.info(f"Model checkpoint saved: {config['save_path']} ({file_size:.2f} MB)")
        else:
            logger.error("Model checkpoint was not saved!")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 