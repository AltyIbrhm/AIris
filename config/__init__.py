"""
Configuration package initialization.
"""
from .schema import validate_config, ConfigError

__all__ = ['validate_config', 'ConfigError'] 