"""
This module contains the configuration classes for AutoGPT.
"""
# from config.ai_config import AIConfig
from config.config import Config,   check_openai_api_key
from config.singleton import AbstractSingleton, Singleton

__all__ = [
    "check_openai_api_key",
    "AbstractSingleton",
    "Config",
    "Singleton",
]
