"""
Device detection and configuration utilities.

Automatically detects available compute devices (CUDA, ROCm, MPS, CPU)
and provides helpers for device management.
"""

import torch
import warnings
from typing import Dict, Optional, Any


def get_device(force_device: Optional[str] = None) -> torch.device:
    """
    Detect and return the best available device.
    
    Priority order:
    1. force_device if specified
    2. CUDA (NVIDIA GPU or AMD ROCm)
    3. MPS (Apple Silicon)
    4. CPU
    
    Args:
        force_device: Force specific device ("cuda", "mps", "cpu", or None for auto)
    
    Returns:
        torch.device: Selected device
    """
    if force_device is not None:
        device_str = force_device.lower()
        if device_str == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        elif device_str == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            warnings.warn("MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_str)
    
    # Auto-detect
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive information about available devices.
    
    Returns:
        Dictionary with device information:
        - device_type: "cuda", "mps", or "cpu"
        - device_name: Human-readable device name
        - device_count: Number of devices (for CUDA)
        - memory_total_gb: Total memory in GB (if available)
        - memory_available_gb: Available memory in GB (if available)
        - cuda_version: CUDA version (if available)
        - pytorch_version: PyTorch version
        - supports_mixed_precision: Whether FP16/BF16 is supported
    """
    info = {
        'device_type': None,
        'device_name': None,
        'device_count': 0,
        'memory_total_gb': None,
        'memory_available_gb': None,
        'cuda_version': None,
        'pytorch_version': torch.__version__,
        'supports_mixed_precision': False
    }
    
    # Check CUDA (NVIDIA or AMD ROCm)
    if torch.cuda.is_available():
        info['device_type'] = 'cuda'
        info['device_count'] = torch.cuda.device_count()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['supports_mixed_precision'] = True
        
        # Memory info
        try:
            mem_total = torch.cuda.get_device_properties(0).total_memory
            mem_reserved = torch.cuda.memory_reserved(0)
            mem_allocated = torch.cuda.memory_allocated(0)
            
            info['memory_total_gb'] = mem_total / (1024**3)
            info['memory_available_gb'] = (mem_total - mem_reserved) / (1024**3)
        except Exception:
            pass
    
    # Check MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['device_type'] = 'mps'
        info['device_name'] = 'Apple Silicon (M-series)'
        info['device_count'] = 1
        # MPS support for FP16 has improved in recent PyTorch versions
        info['supports_mixed_precision'] = False  # Still conservative for stability
    
    # CPU fallback
    else:
        info['device_type'] = 'cpu'
        info['device_name'] = 'CPU'
        info['device_count'] = 1
        info['supports_mixed_precision'] = False
    
    return info


def get_recommended_batch_size(base_batch_size: int = 32, base_memory_gb: float = 16.0) -> int:
    """
    Recommend batch size based on available GPU memory.
    
    Args:
        base_batch_size: Base batch size for base_memory_gb
        base_memory_gb: Reference memory size (default: 16GB)
    
    Returns:
        Recommended batch size
    """
    info = get_device_info()
    
    if info['memory_total_gb'] is None:
        # Can't determine memory, use base
        return base_batch_size
    
    # Scale batch size by memory ratio
    memory_ratio = info['memory_total_gb'] / base_memory_gb
    recommended_batch = max(1, int(base_batch_size * memory_ratio))
    
    # Round to nearest power of 2 for efficiency
    recommended_batch = 2 ** round(torch.log2(torch.tensor(float(recommended_batch))).item())
    
    return min(recommended_batch, 128)  # Cap at 128


def print_device_info():
    """Print formatted device information."""
    info = get_device_info()
    device = get_device()
    
    print("=" * 60)
    print("Device Configuration")
    print("=" * 60)
    print(f"PyTorch version: {info['pytorch_version']}")
    print(f"Device type: {info['device_type']}")
    print(f"Device name: {info['device_name']}")
    
    if info['device_type'] == 'cuda':
        print(f"CUDA version: {info['cuda_version']}")
        print(f"Device count: {info['device_count']}")
        if info['memory_total_gb']:
            print(f"Memory (total): {info['memory_total_gb']:.2f} GB")
            print(f"Memory (available): {info['memory_available_gb']:.2f} GB")
    
    print(f"Mixed precision support: {info['supports_mixed_precision']}")
    print(f"Selected device: {device}")
    
    # Batch size recommendation
    if info['memory_total_gb']:
        recommended_batch = get_recommended_batch_size()
        print(f"\nRecommended batch size: {recommended_batch}")
        print(f"  (Based on {info['memory_total_gb']:.1f}GB memory)")
    
    print("=" * 60)


def check_device_compatibility():
    """
    Check if current device setup is compatible with training requirements.
    
    Prints warnings if device configuration may cause issues.
    """
    info = get_device_info()
    
    print("\nDevice Compatibility Check:")
    print("-" * 40)
    
    # Check 1: Device availability
    if info['device_type'] == 'cpu':
        print("[WARNING] Training on CPU will be very slow")
        print("   Recommendation: Use GPU if available")
    else:
        print(f"[OK] GPU detected: {info['device_name']}")
    
    # Check 2: Memory
    if info['memory_total_gb'] is not None:
        if info['memory_total_gb'] < 4:
            print(f"[WARNING] Low GPU memory ({info['memory_total_gb']:.1f}GB)")
            print("   Recommendation: Use batch_size=4-8 with gradient_accumulation")
        elif info['memory_total_gb'] < 8:
            print(f"[INFO] Moderate GPU memory ({info['memory_total_gb']:.1f}GB)")
            print("   Recommendation: Use batch_size=8-16")
        else:
            print(f"[OK] Sufficient GPU memory ({info['memory_total_gb']:.1f}GB)")
    
    # Check 3: Mixed precision
    if info['supports_mixed_precision']:
        print("[OK] Mixed precision (FP16) supported - can train faster")
    else:
        if info['device_type'] == 'mps':
            print("[INFO] MPS doesn't support FP16 well - will use FP32")
        else:
            print("[INFO] Mixed precision not available")
    
    # Check 4: PyTorch version
    try:
        major, minor = map(int, info['pytorch_version'].split('.')[:2])
        if major < 2:
            print(f"[WARNING] PyTorch version {info['pytorch_version']} is outdated")
            print("   Recommendation: Upgrade to PyTorch 2.0+ for better performance")
        else:
            print(f"[OK] PyTorch version {info['pytorch_version']} is recent")
    except Exception:
        pass
    
    print("-" * 40)


if __name__ == '__main__':
    # Run device check
    print_device_info()
    check_device_compatibility()
    
    # Test device creation
    print("\n" + "=" * 60)
    print("Testing device creation:")
    print("=" * 60)
    
    device = get_device()
    print(f"Auto-detected device: {device}")
    
    # Test tensor creation
    try:
        x = torch.randn(2, 3, 224, 224).to(device)
        print(f"[OK] Successfully created tensor on {device}")
        print(f"  Tensor shape: {x.shape}")
        print(f"  Tensor device: {x.device}")
        
        # Test basic operation
        y = x * 2.0
        print(f"[OK] Successfully performed computation on {device}")
    except Exception as e:
        print(f"[ERROR] Failed to create tensor: {e}")