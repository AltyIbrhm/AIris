"""
Configuration schema validation.
"""
from typing import Dict, Any, List
from pydantic import BaseModel, Field, ValidationError, field_validator

class ConfigError(Exception):
    """Custom exception for configuration validation errors."""
    pass

class ExchangeConfig(BaseModel):
    """Exchange configuration schema."""
    name: str = Field(..., description="Name of the exchange")
    api_key: str = Field(..., description="API key for exchange authentication")
    api_secret: str = Field(..., description="API secret for exchange authentication")

    @field_validator('name')
    @classmethod
    def validate_exchange_name(cls, v):
        valid_exchanges = ['binance', 'kucoin', 'huobi']
        if v.lower() not in valid_exchanges:
            raise ValueError("Invalid exchange name")
        return v.lower()

class TradingConfig(BaseModel):
    """Trading configuration schema."""
    symbols: List[str] = Field(..., description="List of trading symbols")
    timeframe: str = Field(..., description="Trading timeframe")
    max_position_size: float = Field(..., gt=0, description="Maximum position size in USDT")
    max_daily_trades: int = Field(..., gt=0, description="Maximum number of trades per day")

    @field_validator('timeframe')
    @classmethod
    def validate_timeframe(cls, v):
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        if v not in valid_timeframes:
            raise ValueError("Invalid timeframe")
        return v

class RiskConfig(BaseModel):
    """Risk management configuration schema."""
    max_drawdown: float = Field(..., ge=0, lt=1, description="Maximum drawdown allowed")
    stop_loss: float = Field(..., gt=0, lt=1, description="Stop loss percentage")
    take_profit: float = Field(..., gt=0, description="Take profit percentage")

class ModelConfig(BaseModel):
    """Model configuration schema."""
    type: str = Field(..., description="Type of ML model")
    input_dim: int = Field(..., gt=0, description="Input dimension for the model")
    hidden_dim: int = Field(..., gt=0, description="Hidden dimension for the model")
    num_layers: int = Field(..., gt=0, description="Number of layers")
    dropout: float = Field(..., ge=0, lt=1, description="Dropout rate")

    @field_validator('type')
    @classmethod
    def validate_model_type(cls, v):
        valid_types = ['lstm', 'gru', 'transformer']
        if v.lower() not in valid_types:
            raise ValueError("Invalid model type")
        return v.lower()

class Config(BaseModel):
    """Main configuration schema."""
    exchange: ExchangeConfig
    trading: TradingConfig
    risk: RiskConfig
    model: ModelConfig

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration against the schema.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if validation succeeds
        
    Raises:
        ConfigError: If validation fails
    """
    try:
        Config(**config)
        return True
    except ValidationError as e:
        error_messages = ["Invalid configuration"]
        missing_fields = []
        
        for error in e.errors():
            if error['type'] == 'value_error':
                error_messages.append(error['msg'])
            elif error['type'] == 'missing':
                missing_fields.append(error['loc'][-1])
            elif error['type'] == 'greater_than' or error['type'] == 'less_than':
                if 'risk' in error['loc']:
                    error_messages.append("Invalid risk parameters")
                elif 'model' in error['loc']:
                    error_messages.append("Invalid model parameters")
            else:
                error_messages.append(str(error))
        
        if missing_fields:
            error_messages.append("Missing required fields: " + ", ".join(missing_fields))
            
        raise ConfigError("\n".join(error_messages)) 