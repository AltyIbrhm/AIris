import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for time series data.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.3):
        """
        Initialize the LSTM classifier.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden units in LSTM
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 3)  # 3 classes: SELL, HOLD, BUY
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Get the last output of the sequence
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Final classification layer
        logits = self.fc(last_output)
        
        return logits 