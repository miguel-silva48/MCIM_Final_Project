"""
Environment detection utilities for Kaggle vs. Local execution.
"""

from pathlib import Path


def is_kaggle() -> bool:
    """
    Check if code is running on Kaggle.
    
    Returns:
        bool: True if running on Kaggle, False otherwise.
    """
    return Path('/kaggle/input').exists()


def is_local() -> bool:
    """
    Check if code is running locally (not on Kaggle).
    
    Returns:
        bool: True if running locally, False otherwise.
    """
    return not is_kaggle()


def get_execution_env() -> str:
    """
    Get the current execution environment.
    
    Returns:
        str: 'kaggle' if running on Kaggle, 'local' otherwise.
    """
    return 'kaggle' if is_kaggle() else 'local'


if __name__ == '__main__':
    # Quick test
    env = get_execution_env()
    print(f"Execution environment: {env}")
    print(f"Is Kaggle: {is_kaggle()}")
    print(f"Is Local: {is_local()}")
