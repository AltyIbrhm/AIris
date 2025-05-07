import os
from ml.labels.label_generator import process_all_symbols

def main():
    """Generate labels for all available feature files."""
    input_path = "ml/data/processed"
    output_path = "ml/data/processed"
    
    # Process all symbols
    process_all_symbols(
        input_path=input_path,
        output_path=output_path,
        future_bars=10,  # Look ahead 10 bars
        atr_window=14,   # Standard ATR window
        atr_multiplier=0.75  # Conservative threshold
    )

if __name__ == "__main__":
    main() 