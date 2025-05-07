"""
Configuration schema validation.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator, validator
import yaml
import re

class ConfigError(Exception):
    """Custom exception for configuration validation errors."""
    pass

class ExchangeConfig(BaseModel):
    """Exchange configuration schema."""
    name: str = Field(..., description="Exchange name")
    api_key: str = Field(..., description="API key")
    api_secret: str = Field(..., description="API secret")

    @field_validator('name')
    @classmethod
    def validate_exchange_name(cls, v):
        valid_exchanges = ['binance', 'kucoin', 'huobi']
        if v.lower() not in valid_exchanges:
            raise ValueError("Invalid exchange name")
        return v.lower()

class RiskConfig(BaseModel):
    """Risk management configuration."""
    min_confidence: float = Field(0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_open_positions_total: int = Field(3, gt=0, description="Maximum total open positions")
    max_open_positions_per_symbol: int = Field(1, gt=0, description="Maximum positions per symbol")
    max_drawdown_percent: float = Field(10.0, gt=0.0, le=100.0, description="Maximum drawdown percentage")
    max_daily_loss: float = Field(300.0, gt=0.0, description="Maximum daily loss")
    default_sl_percent: float = Field(2.0, gt=0.0, le=100.0, description="Default stop loss percentage")
    default_tp_percent: float = Field(4.0, gt=0.0, le=100.0, description="Default take profit percentage")
    duplicate_signal_block_minutes: int = Field(5, gt=0, description="Minutes to block duplicate signals")
    max_position_size_percent: float = Field(10.0, gt=0.0, le=100.0, description="Maximum position size percentage")
    max_leverage: float = Field(1.0, ge=1.0, description="Maximum leverage")
    risk_free_rate: float = Field(0.02, ge=0.0, le=1.0, description="Risk-free rate")
    volatility_lookback: int = Field(20, gt=0, description="Volatility lookback period")
    position_sizing_method: str = Field('kelly', description="Position sizing method")
    emergency_stop_loss_percent: float = Field(5.0, gt=0.0, le=100.0, description="Emergency stop loss percentage")

    @validator('default_tp_percent')
    def validate_risk_reward(cls, v, values):
        """Ensure take profit is greater than stop loss."""
        if 'default_sl_percent' in values and v <= values['default_sl_percent']:
            raise ValueError("Take profit must be greater than stop loss")
        return v

class ModelConfig(BaseModel):
    """AI model configuration."""
    model_type: str = Field(..., description="Model type")
    input_features: List[str] = Field(..., description="Input features")
    output_features: List[str] = Field(..., description="Output features")
    sequence_length: int = Field(100, gt=0, description="Sequence length")
    batch_size: int = Field(32, gt=0, description="Batch size")
    learning_rate: float = Field(0.001, gt=0.0, description="Learning rate")
    epochs: int = Field(100, gt=0, description="Number of epochs")
    validation_split: float = Field(0.2, gt=0.0, lt=1.0, description="Validation split")

class TradingConfig(BaseModel):
    """Trading system configuration."""
    exchange: str = Field(..., description="Exchange name")
    symbols: List[str] = Field(..., description="Trading symbols")
    interval: str = Field(..., description="Trading interval")
    poll_interval: int = Field(..., gt=0, description="Poll interval in seconds")
    risk_config: RiskConfig = Field(..., description="Risk configuration")
    ai_model_config: ModelConfig = Field(..., description="AI model configuration")
    paper_trading: bool = Field(True, description="Paper trading mode")
    log_level: str = Field('INFO', description="Logging level")

    @validator('symbols')
    def validate_symbols(cls, v):
        """Validate symbol format."""
        for symbol in v:
            if not re.match(r'^[A-Z]+/[A-Z]+$', symbol):
                raise ValueError(f"Invalid symbol format: {symbol}")
        return v

    @validator('interval')
    def validate_interval(cls, v):
        """Validate trading interval."""
        valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
        if v not in valid_intervals:
            raise ValueError(f"Invalid timeframe: {v}")
        return v

class Config(BaseModel):
    """Main configuration schema."""
    exchange: ExchangeConfig = Field(..., description="Exchange configuration")
    trading: TradingConfig = Field(..., description="Trading configuration")
    risk: RiskConfig = Field(..., description="Risk configuration")
    model: ModelConfig = Field(..., description="Model configuration")

    class Config:
        extra = "forbid"

    def __init__(self, **data):
        try:
            super().__init__(**data)
        except ValidationError as e:
            error_messages = ["Invalid configuration"]
            missing_fields = []
            
            for error in e.errors():
                if error['type'] == 'value_error':
                    error_messages.append(error['msg'])
                elif error['type'] == 'missing':
                    missing_fields.append('.'.join(str(x) for x in error['loc']))
                elif error['type'] == 'greater_than' or error['type'] == 'less_than':
                    if 'risk' in error['loc']:
                        error_messages.append(f"Invalid risk parameters: {error['msg']}")
                    elif 'model' in error['loc']:
                        error_messages.append(f"Invalid model parameters: {error['msg']}")
                    else:
                        error_messages.append(error['msg'])
                elif error['type'] == 'model_type':
                    error_messages.append(f"Invalid configuration structure: {error['msg']}")
                elif error['type'] == 'list_type':
                    error_messages.append(f"Invalid list format: {error['msg']}")
                else:
                    error_messages.append(str(error['msg']))
            
            if missing_fields:
                error_messages.append("Missing required fields: " + ", ".join(missing_fields))
                
            raise ConfigError("\n".join(error_messages))

    @classmethod
    def parse_obj(cls, obj: Dict[str, Any]) -> 'Config':
        """Parse and validate configuration object."""
        try:
            return cls(**obj)
        except ValidationError as e:
            error_messages = ["Invalid configuration"]
            missing_fields = []
            
            for error in e.errors():
                if error['type'] == 'value_error':
                    error_messages.append(error['msg'])
                elif error['type'] == 'missing':
                    missing_fields.append('.'.join(str(x) for x in error['loc']))
                elif error['type'] == 'greater_than' or error['type'] == 'less_than':
                    if 'risk' in error['loc']:
                        error_messages.append(f"Invalid risk parameters: {error['msg']}")
                    elif 'model' in error['loc']:
                        error_messages.append(f"Invalid model parameters: {error['msg']}")
                    else:
                        error_messages.append(error['msg'])
                elif error['type'] == 'model_type':
                    error_messages.append(f"Invalid configuration structure: {error['msg']}")
                elif error['type'] == 'list_type':
                    error_messages.append(f"Invalid list format: {error['msg']}")
                else:
                    error_messages.append(str(error['msg']))
            
            if missing_fields:
                error_messages.append("Missing required fields: " + ", ".join(missing_fields))
                
            raise ConfigError("\n".join(error_messages))

def validate_config(config_dict: Dict[str, Any]) -> Config:
    """Validate configuration dictionary."""
    try:
        return Config.parse_obj(config_dict)
    except ValidationError as e:
        error_messages = ["Invalid configuration"]
        missing_fields = []
        
        for error in e.errors():
            if error['type'] == 'value_error':
                error_messages.append(error['msg'])
            elif error['type'] == 'missing':
                missing_fields.append('.'.join(str(x) for x in error['loc']))
            elif error['type'] == 'greater_than' or error['type'] == 'less_than':
                if 'risk' in error['loc']:
                    error_messages.append(f"Invalid risk parameters: {error['msg']}")
                elif 'model' in error['loc']:
                    error_messages.append(f"Invalid model parameters: {error['msg']}")
                else:
                    error_messages.append(error['msg'])
            elif error['type'] == 'model_type':
                error_messages.append(f"Invalid configuration structure: {error['msg']}")
            elif error['type'] == 'list_type':
                error_messages.append(f"Invalid list format: {error['msg']}")
            else:
                error_messages.append(str(error['msg']))
        
        if missing_fields:
            error_messages.append("Missing required fields: " + ", ".join(missing_fields))
            
        raise ConfigError("\n".join(error_messages))

def load_config(config_path: str) -> TradingConfig:
    """Load and validate configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return Config.parse_obj(config_dict).trading
    except Exception as e:
        if isinstance(e, ValidationError):
            raise ConfigError("\n".join([
                "Invalid configuration",
                *[str(err['msg']) for err in e.errors()],
                "Please check your configuration file."
            ]))
        raise ConfigError(f"Error loading config: {str(e)}") 