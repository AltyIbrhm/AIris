"""
ML-based trading strategy using AI models for signal generation.
"""
from typing import Dict, Any
import numpy as np
from .base import BaseStrategy

class AIStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any]):
        """Initialize the AI strategy with model configuration."""
        super().__init__(config)
        self.model_path = config.get('model_path', 'models/trained/lstm_model.h5')
        self.prediction_threshold = config.get('prediction_threshold', 0.6)
        self.model = None  # Will be loaded from models.inference

    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals using ML model predictions."""
        try:
            # Get model prediction
            prediction = self.model.predict(market_data)
            
            # Generate signal based on prediction
            if prediction > self.prediction_threshold:
                action = 'buy'
            elif prediction < -self.prediction_threshold:
                action = 'sell'
            else:
                action = 'hold'
            
            return {
                'action': action,
                'price': market_data['close'][-1],
                'timestamp': market_data['timestamp'][-1],
                'confidence': abs(prediction),
                'prediction': prediction
            }
        except Exception as e:
            print(f"Error generating AI signal: {e}")
            return {
                'action': 'hold',
                'price': market_data['close'][-1],
                'timestamp': market_data['timestamp'][-1],
                'error': str(e)
            } 