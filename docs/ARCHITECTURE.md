# AIris Architecture

## Overview
AIris is a professional-grade AI-driven trading bot built with a modular, scalable architecture. The system is designed to be maintainable, testable, and extensible.

## Core Components

### 1. Core Engine (`core/`)
- **engine.py**: Main trading engine that orchestrates the entire system
- **interfaces.py**: Core interfaces defining system contracts
- **runner.py**: Execution runner for trading operations

### 2. Strategy Module (`strategy/`)
- **base.py**: Abstract base class for all strategies
- **ai_strategy.py**: AI-driven trading strategy
- **ema_strategy.py**: EMA-based technical strategy
- **signal_combiner.py**: Logic for combining multiple strategy signals

### 3. Risk Management (`risk/`)
- **base.py**: Base risk management functionality
- **exposure.py**: Position sizing and exposure management
- **sl_tp.py**: Stop loss and take profit management

### 4. Data Handling (`data/`)
- **fetcher.py**: Market data retrieval
- **preprocessor.py**: Data preprocessing and feature engineering

### 5. ML Models (`models/`)
- LSTM-based prediction models (Phase 3)
- Model training and evaluation utilities

### 6. Configuration (`config/`)
- YAML-based configuration
- Pydantic schema validation
- Environment variable management

### 7. Utilities (`utils/`)
- Logging system
- Common utilities
- Enumeration definitions

## Data Flow
1. Market data is fetched and preprocessed
2. Strategies generate trading signals
3. Risk management evaluates and adjusts positions
4. Core engine executes trades
5. Results are logged and monitored

## Design Principles
- Interface-driven development
- Separation of concerns
- Test-driven development
- Configuration as code
- Comprehensive logging
- Risk-first approach

## Technology Stack
- Python 3.11+
- PyTorch (for ML models)
- Pydantic (for configuration)
- pytest (for testing)
- Rich (for logging) 