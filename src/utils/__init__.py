"""Utility functions for environment detection and data paths."""

from .environment import is_kaggle, is_local, get_execution_env
from .data_paths import get_data_paths
from .device_check import get_device, get_device_info, print_device_info
from .config_loader import load_config
from .checkpoint import CheckpointManager
from .logging_utils import TrainingLogger, MetricsTracker

def print_environment_info():
    """Print information about current execution environment."""
    env = get_execution_env()
    
    print("=" * 70)
    print("ENVIRONMENT INFORMATION")
    print("=" * 70)
    print(f"Environment: {env.upper()}")
    
    if env == 'kaggle':
        print(f"Kaggle input directory: /kaggle/input")
        print(f"Kaggle working directory: /kaggle/working")
        
        # List available datasets
        from pathlib import Path
        kaggle_input = Path('/kaggle/input')
        if kaggle_input.exists():
            datasets = list(kaggle_input.iterdir())
            print(f"\nAvailable datasets ({len(datasets)}):")
            for dataset in datasets[:5]:  # Show first 5
                print(f"  - {dataset.name}")
            if len(datasets) > 5:
                print(f"  ... and {len(datasets) - 5} more")
    else:
        from pathlib import Path
        print(f"Working directory: {Path.cwd()}")
    
    print("=" * 70)

def get_output_path(project_root=None):
    """Get output directory based on execution environment."""
    from pathlib import Path
    env = get_execution_env()
    
    if env == 'kaggle':
        return Path('/kaggle/working/outputs')
    else:
        if project_root is None:
            project_root = Path(__file__).parents[2]
        return project_root / 'outputs' / 'training_runs'

__all__ = [
    'is_kaggle',
    'is_local',
    'get_execution_env',
    'get_data_paths',
    'get_device',
    'get_device_info',
    'print_device_info',
    'print_environment_info',
    'get_output_path',
    'load_config',
    'CheckpointManager',
    'TrainingLogger',
    'MetricsTracker'
]
