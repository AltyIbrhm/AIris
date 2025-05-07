import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from ml.data.dataset_builder import TimeSeriesDataset
from ml.models.lstm_model import LSTMClassifier
import yaml
import os
import logging
from typing import Dict, Any
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

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

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, save_dir):
    """Plot training metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_metrics.png')
    plt.close()

def analyze_hold_misclassifications(y_true: np.ndarray, y_pred: np.ndarray, features: np.ndarray, indices: np.ndarray) -> None:
    """
    Analyze misclassified HOLD samples to understand patterns.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        features: Feature matrix
        indices: Sample indices
    """
    # Create DataFrame for analysis
    hold_analysis = pd.DataFrame({
        'index': indices,
        'true_label': y_true,
        'predicted_label': y_pred,
        'is_hold': y_true == 1,
        'is_misclassified': (y_true == 1) & (y_pred != 1)
    })
    
    # Calculate statistics for misclassified HOLD samples
    logger.info("\n=== HOLD Misclassification Analysis ===")
    logger.info(f"Total HOLD samples: {len(hold_analysis[hold_analysis['is_hold']])}")
    logger.info(f"Misclassified HOLD samples: {len(hold_analysis[hold_analysis['is_misclassified']])}")
    
    # Analyze where HOLD is being confused
    confusion_counts = hold_analysis[hold_analysis['is_misclassified']]['predicted_label'].value_counts()
    logger.info("\nHOLD misclassified as:")
    for label, count in confusion_counts.items():
        percentage = count / len(hold_analysis[hold_analysis['is_misclassified']]) * 100
        logger.info(f"Class {label}: {count} samples ({percentage:.1f}%)")
    
    # Save detailed analysis to CSV
    output_path = Path("ml/evaluation/hold_analysis")
    output_path.mkdir(parents=True, exist_ok=True)
    hold_analysis.to_csv(output_path / "hold_misclassifications.csv", index=False)
    logger.info(f"\nDetailed analysis saved to {output_path}/hold_misclassifications.csv")

def train_model() -> None:
    """
    Main training function that handles:
    - Dataset loading and splitting
    - Model initialization
    - Training loop with validation
    - Model checkpointing
    """
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Create output directory
        output_dir = Path("ml/train/figures")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

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
        train_size = int(len(dataset) * (1 - config["val_split"]))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Calculate class weights for sampler using only training data
        train_labels = [dataset.labels["label"].iloc[idx] for idx in train_dataset.indices]
        class_sample_count = np.array([np.sum(np.array(train_labels) == t) for t in [0, 1, 2]])
        weights = 1. / class_sample_count
        samples_weight = np.array([weights[t] for t in train_labels])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            sampler=sampler
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False
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
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config["lr_factor"],
            patience=config["lr_patience"],
            min_lr=config["min_lr"]
        )
        
        # Create checkpoint directory
        os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
        
        # Training loop
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
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
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config["gradient_clip"]
                )
                
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
            train_losses.append(avg_train_loss)
            train_accs.append(train_accuracy)

            # Validation phase
            model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            val_preds = []
            val_true = []
            val_features = []
            
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    val_logits = model(X)
                    val_loss += criterion(val_logits, y).item()
                    
                    _, predicted = torch.max(val_logits.data, 1)
                    total_val += y.size(0)
                    correct_val += (predicted == y).sum().item()
                    
                    # Store predictions and features for analysis
                    val_preds.extend(predicted.cpu().numpy())
                    val_true.extend(y.cpu().numpy())
                    val_features.extend(X.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct_val / total_val
            val_losses.append(avg_val_loss)
            val_accs.append(val_accuracy)

            # Update learning rate
            scheduler.step(avg_val_loss)

            # Logging
            logger.info(
                f"[Epoch {epoch+1}/{config['epochs']}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.2f}%"
            )
            
            # Print classification report for validation set
            report = classification_report(
                val_true,
                val_preds,
                target_names=["SELL", "HOLD", "BUY"],
                digits=3
            )
            logger.info("\nValidation Classification Report:")
            logger.info(report)

            # Analyze HOLD misclassifications
            if epoch % 5 == 0:  # Analyze every 5 epochs
                analyze_hold_misclassifications(
                    np.array(val_true),
                    np.array(val_preds),
                    np.array(val_features),
                    np.array(range(len(val_true)))  # Using indices instead of timestamps
                )

            # Model checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, config["save_path"])
                
                logger.info(f"✅ New best model saved: {config['save_path']}")
            else:
                patience_counter += 1
                
            # Early stopping check
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Plot training metrics
        plot_training_metrics(
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            output_dir
        )
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