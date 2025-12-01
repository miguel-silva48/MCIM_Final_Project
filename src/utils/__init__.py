"""Utility functions for environment detection and data paths."""

from .environment import is_kaggle, is_local, get_execution_env
from .data_paths import get_data_paths

__all__ = ['is_kaggle', 'is_local', 'get_execution_env', 'get_data_paths']
