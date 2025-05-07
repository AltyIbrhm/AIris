import pandas as pd
import numpy as np
import os
import logging
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
RAW_PATH = "ml/data/raw"
SAVE_PATH = "ml/data/processed"

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features based on their type.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        
    Returns:
        pd.DataFrame: DataFrame with normalized features
    """
    df = df.copy()
    
    # Features that should be normalized to mean 0, std 1
    zscore_features = [
        'ema_8', 'ema_21', 'ema_ratio',
        'macd_line', 'macd_signal',
        'bb_width', 'bb_zscore',
        'momentum_10', 'rate_of_change_5',
        'volume_zscore_10'
    ]
    
    # Features that should be min-max scaled to [0, 1]
    minmax_features = [
        'bb_b', 'high_low_pct',
        'atr_14', 'atr_pct'
    ]
    
    # Features that should be kept in their original range
    original_range_features = [
        'rsi_14', 'rsi_14_lag1',  # RSI is already 0-100
        'obv', 'obv_lag1',  # OBV should be kept in original range
        'macd_hist', 'macd_hist_lag1',  # MACD histogram should be kept in original range
        'log_return_1', 'log_return_1_lag1',  # Log returns should be kept in original range
        'log_return_5',  # Log returns should be kept in original range
        # Regime flags should be kept in their original range (0, 1, or 0-2)
        'is_trending', 'high_volatility', 'low_volatility',
        'is_ranging', 'price_above_bb_mid', 'rsi_regime'
    ]
    
    # Apply z-score normalization
    if zscore_features:
        scaler = StandardScaler()
        df[zscore_features] = scaler.fit_transform(df[zscore_features])
    
    # Apply min-max scaling
    if minmax_features:
        scaler = MinMaxScaler()
        df[minmax_features] = scaler.fit_transform(df[minmax_features])
    
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate technical indicators and features from OHLCV data.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    df = df.copy()
    
    try:
        # Trend Indicators
        # EMA
        df["ema_8"] = EMAIndicator(df["close"], window=8).ema_indicator()
        df["ema_21"] = EMAIndicator(df["close"], window=21).ema_indicator()
        df["ema_ratio"] = df["ema_8"] / df["ema_21"]
        
        # RSI
        df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
        
        # MACD
        macd = MACD(close=df["close"])
        df["macd_line"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close=df["close"], window=20, window_dev=2)
        df["bb_b"] = bb.bollinger_pband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_zscore"] = (df["close"] - bb.bollinger_mavg()) / bb.bollinger_hband()
        
        # Volatility & Risk
        # ATR
        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
        df["atr_14"] = atr.average_true_range()
        df["atr_pct"] = df["atr_14"] / df["close"]  # This will be min-max scaled
        df["high_low_pct"] = (df["high"] - df["low"]) / df["close"]
        
        # Momentum
        df["momentum_10"] = df["close"].diff(10)
        df["rate_of_change_5"] = df["close"].pct_change(5)
        df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
        df["log_return_5"] = np.log(df["close"] / df["close"].shift(5))
        
        # Volume & Context
        df["volume_zscore_10"] = (df["volume"] - df["volume"].rolling(10).mean()) / df["volume"].rolling(10).std()
        df["obv"] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
        
        # --- Add Lagged Features ---
        df["rsi_14_lag1"] = df["rsi_14"].shift(1)
        df["macd_hist_lag1"] = df["macd_hist"].shift(1)
        df["log_return_1_lag1"] = df["log_return_1"].shift(1)
        df["obv_lag1"] = df["obv"].shift(1)
        
        # Clean NaN values before calculating regime flags
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # --- Regime Flags ---
        # 1. Trend Regime (EMA crossover)
        df["is_trending"] = (df["ema_8"] > df["ema_21"]).astype(int)
        
        # 2. Volatility Regime (ATR % thresholds)
        atr_q75 = df["atr_pct"].quantile(0.75)
        atr_q25 = df["atr_pct"].quantile(0.25)
        df["high_volatility"] = (df["atr_pct"] > atr_q75).astype(int)
        df["low_volatility"] = (df["atr_pct"] < atr_q25).astype(int)
        
        # 3. Ranging Market (Narrow Bollinger Band width)
        bb_q25 = df["bb_width"].quantile(0.25)
        df["is_ranging"] = (df["bb_width"] < bb_q25).astype(int)
        
        # 4. Price above Bollinger Middle Band
        df["bb_mavg"] = (df["bb_b"] - 0.5)  # Approximates middle position
        df["price_above_bb_mid"] = (df["bb_mavg"] > 0).astype(int)
        
        # 5. RSI Regime
        # Fill NaN values with 50 (neutral) before creating regimes
        df["rsi_14"] = df["rsi_14"].fillna(50)
        df["rsi_regime"] = pd.cut(df["rsi_14"], bins=[0, 30, 70, 100], labels=[0, 1, 2]).astype(int)
        
        # Replace infinite values with NaN and then forward fill
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Store original values before normalization
        original_values = {
            'rsi_14': df['rsi_14'].copy(),
            'rsi_14_lag1': df['rsi_14_lag1'].copy(),
            'obv': df['obv'].copy(),
            'obv_lag1': df['obv_lag1'].copy(),
            'macd_hist': df['macd_hist'].copy(),
            'macd_hist_lag1': df['macd_hist_lag1'].copy(),
            'log_return_1': df['log_return_1'].copy(),
            'log_return_1_lag1': df['log_return_1_lag1'].copy(),
            'log_return_5': df['log_return_5'].copy()
        }
        
        # Normalize features
        df = normalize_features(df)
        
        # Restore original values
        for key, value in original_values.items():
            df[key] = value
        
        # Drop intermediate calculation columns
        df.drop(columns=["bb_mavg"], inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error generating features: {str(e)}")
        raise

def process_all_symbols():
    """
    Process all symbol data files in the raw data directory.
    """
    try:
        os.makedirs(SAVE_PATH, exist_ok=True)
        
        for file in os.listdir(RAW_PATH):
            if not file.endswith(".csv"):
                continue
                
            symbol = file.split("_")[0]
            logger.info(f"Processing {symbol}...")
            
            # Read and prepare data
            df = pd.read_csv(os.path.join(RAW_PATH, file))
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.astype({
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float
            })
            
            # Generate features
            df = generate_features(df)
            
            # Save processed data
            output_file = os.path.join(SAVE_PATH, f"{symbol}_5m_features.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"[✓] Saved features for {symbol} to {output_file}")
            
    except Exception as e:
        logger.error(f"Error processing symbols: {str(e)}")
        raise

if __name__ == "__main__":
    process_all_symbols() 