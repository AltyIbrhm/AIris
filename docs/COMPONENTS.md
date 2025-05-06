# AIris Components

## Core Components

### Trading Engine
The core engine (`core/engine.py`) is the central orchestrator of the trading system. It:
- Manages the trading lifecycle
- Coordinates between strategies and risk management
- Handles order execution
- Maintains system state

### Strategy Module
The strategy module (`strategy/`) implements different trading strategies:

#### Base Strategy (`base.py`)
- Abstract base class defining strategy interface
- Common functionality for all strategies
- Signal generation framework

#### AI Strategy (`ai_strategy.py`)
- Machine learning-based trading signals
- Model integration points
- Feature engineering pipeline

#### EMA Strategy (`ema_strategy.py`)
- Technical analysis based on EMA
- Signal generation logic
- Parameter optimization

#### Signal Combiner (`signal_combiner.py`)
- Combines signals from multiple strategies
- Weighted signal aggregation
- Conflict resolution

### Risk Management
The risk module (`risk/`) handles all risk-related aspects:

#### Base Risk (`base.py`)
- Risk calculation framework
- Position sizing logic
- Risk metrics

#### Exposure Management (`exposure.py`)
- Position sizing
- Portfolio exposure limits
- Leverage management

#### Stop Loss/Take Profit (`sl_tp.py`)
- Dynamic SL/TP calculation
- Risk-reward optimization
- Position management

### Data Handling
The data module (`data/`) manages market data:

#### Data Fetcher (`fetcher.py`)
- Market data retrieval
- Data source integration
- Real-time data handling

#### Preprocessor (`preprocessor.py`)
- Data cleaning
- Feature engineering
- Technical indicator calculation

### Configuration
The config module (`config/`) handles system configuration:

#### Configuration Schema (`schema.py`)
- Pydantic models for validation
- Configuration structure
- Default values

#### YAML Configuration (`config.yaml`)
- System parameters
- Strategy settings
- Risk parameters

### Utilities
The utils module (`utils/`) provides common functionality:

#### Logger (`logger.py`)
- Rich console output
- File logging
- Log rotation

#### Common Utilities (`common.py`)
- Helper functions
- Type definitions
- Constants

#### Enums (`enums.py`)
- System-wide enumerations
- Status codes
- Trading states

## Integration Points

### Strategy Integration
- Strategies implement the base strategy interface
- Signal generation follows a standard format
- Risk parameters are configurable

### Risk Integration
- Risk management hooks into the trading engine
- Position sizing is dynamic
- SL/TP levels are automatically calculated

### Data Integration
- Data fetcher provides a unified interface
- Preprocessor prepares data for strategies
- Real-time updates are supported

## Testing
- Unit tests for each component
- Integration tests for workflows
- Mock data and configurations 