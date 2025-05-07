class ConfidenceFilter:
    def __init__(self, threshold: float = 0.85):
        """
        Initialize Confidence Filter
        
        Args:
            threshold (float): Minimum confidence threshold for trade entry
        """
        self.threshold = threshold
        
    def check_entry(self, confidence: float, signal: str) -> bool:
        """
        Check if confidence level meets minimum threshold for entry
        
        Args:
            confidence (float): Softmax confidence value (0.0 to 1.0)
            signal (str): Trade signal ("BUY" or "SELL")
            
        Returns:
            bool: True if confidence meets threshold and signal is valid, False otherwise
        """
        if signal not in ["BUY", "SELL"]:
            return False
            
        if not 0 <= confidence <= 1:
            return False
            
        return confidence > self.threshold
    
    def __str__(self) -> str:
        return f"ConfidenceFilter(threshold={self.threshold})" 