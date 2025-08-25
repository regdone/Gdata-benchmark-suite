#!/bin/bash
set -e

echo "[Gdata Benchmark Suite] Starting full setup on Ubuntu 24.04..."

### 1. Update system
sudo apt update && sudo apt upgrade -y

### 2. Install essentials
sudo apt install -y build-essential git wget curl python3 python3-venv python3-pip

### 3. Install NVIDIA Driver (latest stable from CUDA repo)
echo "[Step] Installing NVIDIA Driver + CUDA Toolkit..."
CUDA_REPO_KEY="/etc/apt/keyrings/cuda-archive-keyring.gpg"

if [ ! -f "$CUDA_REPO_KEY" ]; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
fi

sudo apt update
sudo apt install -y cuda nvidia-gds

# Add CUDA to PATH & LD_LIBRARY_PATH (for future sessions)
if ! grep -q "cuda" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi
source ~/.bashrc

### 4. Verify driver + CUDA
echo "[Check] NVIDIA Driver & CUDA"
nvidia-smi || { echo "NVIDIA driver not working. Reboot may be required."; }
nvcc --version || echo "CUDA Toolkit not found (check PATH)."

### 5. Setup Python virtual environment
echo "[Step] Python venv setup..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

### 6. Install Python dependencies
echo "[Step] Installing Python requirements..."
pip install -r requirements.txt

### 7. Install PyTorch with CUDA (latest compatible)
echo "[Step] Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

### 8. Install TensorFlow (latest GPU-enabled)
echo "[Step] Installing TensorFlow..."
pip install tensorflow

### 9. Done
echo "[Gdata Benchmark Suite] Setup complete!"
echo "👉 Activate environment: source venv/bin/activate"
echo "👉 Run benchmarks: ./benchmark_all.sh"
