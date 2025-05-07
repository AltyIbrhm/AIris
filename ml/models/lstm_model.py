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
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Batch normalization for input
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Use bidirectional LSTM
        )
        
        # Batch normalization after LSTM
        self.lstm_bn = nn.BatchNorm1d(hidden_dim * 2)  # *2 for bidirectional
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers with dropout and batch norm
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 3)  # 3 classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Input shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # Apply input batch normalization
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = self.input_bn(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, input_dim)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        context = torch.bmm(attention_weights.transpose(1, 2), lstm_out)  # (batch_size, 1, hidden_dim * 2)
        context = context.squeeze(1)  # (batch_size, hidden_dim * 2)
        
        # Apply batch norm to LSTM output
        context = self.lstm_bn(context)
        
        # Fully connected layers
        out = self.fc1(context)
        out = self.fc1_bn(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out 