# AIris 🤖

[![Tests](https://github.com/yourusername/AIris/actions/workflows/test.yml/badge.svg)](https://github.com/yourusername/AIris/actions/workflows/test.yml)
[![Coverage](https://codecov.io/gh/yourusername/AIris/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/AIris)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

AIris is a professional-grade AI-driven trading bot built with a modular, scalable architecture. The system is designed to be maintainable, testable, and extensible.

## 🏗️ Architecture

AIris follows a modular architecture with clear separation of concerns:

- **Core Engine**: Trading lifecycle management and execution
- **Strategy Module**: Multiple trading strategies (AI, EMA, etc.)
- **Risk Management**: Position sizing, SL/TP, exposure control
- **Data Handling**: Market data fetching and preprocessing
- **ML Models**: LSTM-based prediction models (Phase 3)
- **Configuration**: YAML-based with Pydantic validation
- **Utilities**: Logging, common functions, enums

For detailed architecture documentation, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 🚀 Current Phase

We are currently in Phase 3: AI Integration, focusing on implementing machine learning capabilities and advanced trading strategies. Recent progress includes:

### Completed Components:
- Label Generation System ✅
  - ATR-based dynamic thresholding
  - BUY/SELL/HOLD classification
  - Balanced class distribution
  - Comprehensive test coverage
  - Batch processing support

### In Progress:
- Historical data collection
- Feature engineering pipeline
- Model architecture design

See [ROADMAP.md](docs/ROADMAP.md) for the complete development plan.

## 🛠️ Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AIris.git
cd AIris
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## 🧪 Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=.
```

## 📚 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Component Details](docs/COMPONENTS.md)
- [Development Roadmap](docs/ROADMAP.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors
- Inspired by various open-source trading systems
- Built with modern Python best practices 