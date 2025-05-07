"""
Setup file for the AIris trading bot package.
"""
from setuptools import setup, find_packages

setup(
    name="airis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.0.0",
        "mypy>=1.0.0",
        "pyyaml>=6.0.0",
        "ccxt>=4.0.0",
        "websockets>=11.0.0",
        "aiohttp>=3.8.0",
        "python-binance>=1.0.0"
    ],
    python_requires=">=3.8",
) 