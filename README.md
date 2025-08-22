# Gdata Benchmark Suite

A lightweight benchmark suite to demonstrate **GPU vs CPU performance** on real-world AI/ML workloads.  
Designed for demo, customer evaluation, and quick benchmarking of **NVIDIA RTX 5880 Ada** or similar GPUs.

---

## 📂 Included Benchmarks

1. **Matrix Multiplication (HPC baseline)**
   - `Matrix_Multiplication_Benchmark_CPU.py`
   - `Matrix_Multiplication_Benchmark_GPU.py`

2. **Deep Learning (Vision / Image Classification)**
   - `Training_CIFAR10_CNN_CPU.py`
   - `Training_CIFAR10_CNN_GPU.py`

3. **Stress Test (Memory allocation)**
   - `Stress_Test_GPU_Memory_CPU.py`
   - `Stress_Test_GPU_Memory_GPU.py`

4. **Real AI Workload (NLP inference with Hugging Face Transformers)**
   - `Real_AI_Workload_NLP_CPU.py`
   - `Real_AI_Workload_NLP_GPU.py`

---

## ⚙️ Installation

### 1. Clone repo
```bash
git clone https://github.com/Gdata/Gdata-benchmark-suite.git
cd Gdata-benchmark-suite

### Setup Python environment
We recommend Python ≥ 3.10. On Linux/WSL2/Windows Server:

### 2. Setup Python environment
We recommend Python ≥ 3.10. On Linux/WSL2/Windows Server:
``` bash
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

For PyTorch GPU, ensure you install the correct CUDA-enabled wheel.
See https://pytorch.org/get-started/locally/

## 🚀 Running Benchmarks

Run all benchmarks at once:
``` bash
./benchmark_all.sh
Results will be saved in ./results/benchmark_YYYYMMDD_HHMMSS.log.

Run a specific benchmark:
``` bash
# Example: GEMM on GPU with 16K matrix
MATRIX_N=16384 python3 Matrix_Multiplication_Benchmark_GPU.py

# Example: CIFAR10 CNN on GPU for 5 epochs
EPOCHS=5 BATCH=256 python3 Training_CIFAR10_CNN_GPU.py

## 📊 Output Example
``` bash
>>> Running Matrix_Multiplication_Benchmark_GPU.py
Run 1/3: 5.213 s | 17000.5 GFLOP/s
Run 2/3: 5.210 s | 17020.1 GFLOP/s
AVG perf: 17010 GFLOP/s (GPU)

## 🖥️ Windows Notes
On Windows Server 2022 with GPU passthrough:

- Install CUDA Toolkit + cuDNN: https://developer.nvidia.com/cuda-downloads
- Use PowerShell or Git Bash to run scripts
- Replace ./benchmark_all.sh with direct Python runs, or use benchmark_all.bat (optional)

## 📚 References
- TensorFlow GEMM micro-benchmarks (used in MLPerf & blogs)
- Keras CIFAR10 CNN official example: https://keras.io/examples/vision/cifar10_cnn/
- Hugging Face Transformers sentiment-analysis pipeline
- PyTorch memory stress patterns (community tests for max batch size)

