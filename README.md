# GDATA Benchmark Suite
Bộ công cụ benchmark CPU vs GPU dành cho khách hàng của Gdata. Giúp đo lường và so sánh hiệu năng với các workload thực tế như Deep Learning (MNIST), Matrix Multiplication, Stress Test GPU Memory và AI Workload giả lập.

# Nội dung Repo
Gdata-benchmark-suite/ 
├── README.md 
├── requirements.txt 
├── benchmark_all.sh 
├── Matrix_Multiplication_Benchmark_CPU.py 
├── Matrix_Multiplication_Benchmark_GPU.py 
├── Training_MNIST_DeepLearning_CPU.py 
├── Training_MNIST_DeepLearning_GPU.py 
├── Stress_Test_GPU_Memory_CPU.py 
├── Stress_Test_GPU_Memory_GPU.py 
├── Real_AI_Workload_CPU.py 
├── Real_AI_Workload_GPU.py

# Chuẩn bị môi trường
1. Clone repo 
git clone https://github.com/Gdata/benchmark-suite.git 
cd benchmark-suite

2. Tạo môi trường Python 
Khuyến nghị sử dụng venv hoặc conda. 
Cách 1: venv (mặc định có sẵn trong Python) 
python3 -m venv venv 
source venv/bin/activate 

Cách 2: conda (nếu dùng Miniconda/Anaconda) 
conda create -n benchmark python=3.12 -y 
conda activate benchmark
3. Cài dependencies 
pip install -r requirements.txt

# Chạy Benchmark
Chạy toàn bộ benchmark 
./benchmark_all.sh
Tự động chạy 8 script (CPU & GPU).
Kết quả được lưu vào file benchmark_results.log.

Chạy từng benchmark riêng lẻ 
# GEMM
MATRIX_N=16384 python3 Matrix_Multiplication_Benchmark_GPU.py
MATRIX_N=8192  python3 Matrix_Multiplication_Benchmark_CPU.py

# CIFAR-10
EPOCHS=5 BATCH=256 python3 Training_CIFAR10_CNN_GPU.py
EPOCHS=5 BATCH=128 python3 Training_CIFAR10_CNN_CPU.py

# VRAM stress
CHUNK_MB=512 MAX_GB_HINT=80 python3 Stress_Test_GPU_Memory_GPU.py
ALLOC_GB=16 CHUNK_MB=256 python3 Stress_Test_GPU_Memory_CPU.py

# NLP
SAMPLES=20000 BATCH=64 USE_FP16=1 python3 Real_AI_Workload_NLP_GPU.py
SAMPLES=10000 BATCH=32          python3 Real_AI_Workload_NLP_CPU.py


# Kết quả Benchmark
File benchmark_results.log sẽ chứa: 
- Thời gian chạy (CPU vs GPU) 
- Độ chính xác (accuracy, loss) khi train Deep Learning 
- Thông tin stress test GPU Memory 

Ví dụ output:

> Running Matrix_Multiplication_Benchmark_GPU.py  
CPU time: 1.55s  
GPU time: 0.25s  

# Hỗ trợ Windows Machine
Các script .py chạy bình thường trên Windows VM (với Python + TensorFlow). 
Với Windows, thay vì benchmark_all.sh, có thể chạy thủ công từng script. 
Có thể bổ sung file batch benchmark_all.bat nếu cần chạy tự động.

# Yêu cầu hệ thống
Python 3.10+ (tested on 3.12, 3.13) 
TensorFlow 2.20.0 
GPU NVIDIA + driver CUDA/cuDNN tương ứng (nếu muốn chạy GPU mode)

© 2025 Gdata – Benchmark Suite
