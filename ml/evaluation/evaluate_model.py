import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from pathlib import Path

from ml.data.dataset_builder import TimeSeriesDataset
from ml.models.lstm_model import LSTMClassifier
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(path="config/model_config.yaml"):
    """Load model configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)

def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["SELL", "HOLD", "BUY"],
                yticklabels=["SELL", "HOLD", "BUY"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")

def evaluate(config_path="config/model_config.yaml"):
    """Evaluate the trained model on test data."""
    try:
        # Load configuration
        config = load_config(config_path)
        logger.info("Configuration loaded successfully")

        # Create output directory
        output_dir = Path("ml/evaluation/figures")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load test dataset
        logger.info("Loading test dataset...")
        full_dataset = TimeSeriesDataset(
            config["features_path"],
            config["labels_path"],
            config["seq_len"]
        )
        
        # Split into test set
        val_size = int(len(full_dataset) * config["val_split"])
        test_dataset = torch.utils.data.Subset(
            full_dataset,
            range(len(full_dataset) - val_size, len(full_dataset))
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False
        )
        logger.info(f"Test dataset size: {len(test_dataset)}")

        # Load model
        logger.info("Loading trained model...")
        input_dim = full_dataset[0][0].shape[1]
        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        )
        
        # Load model weights
        model_path = Path(config["save_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load model with weights_only=False for backward compatibility
        checkpoint = torch.load(model_path, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        logger.info("Model loaded successfully")

        # Evaluation
        logger.info("Starting evaluation...")
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for X, y in test_loader:
                logits = model(X)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.numpy())
                all_labels.extend(y.numpy())
                all_probs.extend(probs.numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Print classification report
        logger.info("\n📊 Classification Report:")
        report = classification_report(
            all_labels,
            all_preds,
            target_names=["SELL", "HOLD", "BUY"],
            digits=3
        )
        print(report)

        # Generate and save confusion matrix
        logger.info("Generating confusion matrix...")
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")

        # Calculate additional metrics
        accuracy = np.mean(all_preds == all_labels)
        logger.info(f"\nOverall Accuracy: {accuracy:.3f}")

        # Calculate class-wise metrics
        for i, class_name in enumerate(["SELL", "HOLD", "BUY"]):
            class_mask = all_labels == i
            class_accuracy = np.mean(all_preds[class_mask] == all_labels[class_mask])
            logger.info(f"{class_name} Accuracy: {class_accuracy:.3f}")

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            "predictions": all_preds,
            "probabilities": all_probs,
            "true_labels": all_labels
        }

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate() 