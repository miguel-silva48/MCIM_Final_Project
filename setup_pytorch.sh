#!/bin/bash
# =============================================================================
# PyTorch Installation Script for Multi-Device Support
# Medical Image Captioning Project
# =============================================================================
# Supports: NVIDIA CUDA, AMD ROCm, Apple Silicon (M1/M2/M3/M4), CPU-only
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "PyTorch Multi-Device Installation Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS
OS_TYPE=$(uname)
echo -e "\n${GREEN}Detected OS:${NC} $OS_TYPE"

# Function to install for Mac M1/M2/M3/M4 (Apple Silicon)
install_macos() {
    echo -e "\n${GREEN}Installing PyTorch for Apple Silicon...${NC}"
    echo "Using Metal Performance Shaders (MPS) backend"
    
    pip install --upgrade torch torchvision
    
    echo -e "${GREEN}[SUCCESS] PyTorch installed for Apple Silicon${NC}"
    echo "  Device will be detected as 'mps'"
}

# Function to install for NVIDIA GPU (CUDA)
install_nvidia_cuda() {
    echo -e "\n${GREEN}Installing PyTorch for NVIDIA GPU...${NC}"
    
    # Check CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
        echo "Detected CUDA version: $CUDA_VERSION"
        
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        
        if [[ $CUDA_MAJOR -eq 11 ]]; then
            echo "Installing PyTorch with CUDA 11.8 support"
            pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
        else
            echo "Installing PyTorch with latest CUDA support (default: 12.8)"
            pip install --upgrade torch torchvision
        fi
    else
        echo "CUDA compiler not found, installing latest CUDA build (default: 12.8)"
        pip install --upgrade torch torchvision
    fi
    
    echo -e "${GREEN}[SUCCESS] PyTorch installed for NVIDIA CUDA${NC}"
    echo "  Device will be detected as 'cuda'"
}

# Function to install for AMD GPU (ROCm)
install_amd_rocm() {
    echo -e "\n${GREEN}Installing PyTorch for AMD GPU (ROCm 6.4)...${NC}"
    
    # Install PyTorch with ROCm support (latest available on PyTorch)
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
    
    echo -e "${GREEN}[SUCCESS] PyTorch installed for AMD ROCm${NC}"
    echo "  Device will be detected as 'cuda' (ROCm uses same API)"
}

# Function to install CPU-only version
install_cpu() {
    echo -e "\n${YELLOW}Installing PyTorch (CPU-only version)...${NC}"
    
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
    echo -e "${GREEN}[SUCCESS] PyTorch installed (CPU-only)${NC}"
    echo "  Device will be detected as 'cpu'"
}

# Function to install other training dependencies
install_training_deps() {
    echo -e "\n${GREEN}Installing training dependencies...${NC}"
    
    # Note: These dependencies overlap with requirements_eda.txt
    # Using --upgrade ensures compatibility if EDA requirements are already installed
    
    # Core ML/NLP dependencies
    pip install --upgrade nltk
    pip install --upgrade rouge-score
    pip install --upgrade scikit-learn
    pip install --upgrade pandas
    pip install --upgrade numpy
    
    # Image processing
    pip install --upgrade Pillow
    
    # Progress bars and visualization
    pip install --upgrade tqdm
    pip install --upgrade matplotlib
    
    # Optional: Advanced logging (uncomment if needed)
    # pip install wandb
    # pip install tensorboard
    
    echo -e "${GREEN}[SUCCESS] Training dependencies installed${NC}"
}

# Function to verify installation
verify_installation() {
    echo -e "\n${GREEN}Verifying PyTorch installation...${NC}"
    
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Apple Silicon) available: True')
else:
    print('Running on CPU')
" 2>/dev/null && echo -e "${GREEN}[SUCCESS] Verification passed${NC}" || echo -e "${RED}[ERROR] Verification failed${NC}"
}

# Main installation logic
main() {
    echo -e "\nSelect your device configuration:"
    echo "1) NVIDIA GPU (CUDA)"
    echo "2) AMD GPU (ROCm)"
    echo "3) Apple Silicon M1/M2/M3/M4 (MPS)"
    echo "4) CPU only"
    echo "5) Auto-detect (recommended)"
    echo ""
    read -p "Enter choice [1-5]: " choice
    
    case $choice in
        1)
            install_nvidia_cuda
            ;;
        2)
            install_amd_rocm
            ;;
        3)
            install_macos
            ;;
        4)
            install_cpu
            ;;
        5)
            echo -e "\n${GREEN}Auto-detecting device...${NC}"
            
            if [[ "$OS_TYPE" == "Darwin" ]]; then
                # Mac - check for Apple Silicon
                ARCH=$(uname -m)
                if [[ "$ARCH" == "arm64" ]]; then
                    echo "Detected: Apple Silicon (M-series)"
                    install_macos
                else
                    echo "Detected: Intel Mac (CPU-only)"
                    install_cpu
                fi
            elif [[ "$OS_TYPE" == "Linux" ]]; then
                # Linux - check for GPU
                if command -v nvidia-smi &> /dev/null; then
                    echo "Detected: NVIDIA GPU"
                    install_nvidia_cuda
                elif lspci 2>/dev/null | grep -i 'amd.*vga' &> /dev/null; then
                    echo "Detected: AMD GPU"
                    read -p "Install ROCm support? (y/n): " rocm_choice
                    if [[ "$rocm_choice" == "y" || "$rocm_choice" == "Y" ]]; then
                        install_amd_rocm
                    else
                        install_cpu
                    fi
                else
                    echo "No GPU detected"
                    install_cpu
                fi
            else
                echo -e "${YELLOW}Unknown OS, installing CPU version${NC}"
                install_cpu
            fi
            ;;
        *)
            echo -e "${RED}[ERROR] Invalid choice${NC}"
            exit 1
            ;;
    esac
    
    # Install common training dependencies
    install_training_deps
    
    # Verify installation
    verify_installation
    
    echo -e "\n${GREEN}=========================================="
    echo "Installation Complete!"
    echo "==========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Test PyTorch: python3 -c 'import torch; print(torch.__version__)'"
    echo "  2. Check device: python3 -m src.utils.device_check"
    echo "  3. Run training: jupyter notebook notebooks/03_training.ipynb"
    echo ""
}

# Run main function
main