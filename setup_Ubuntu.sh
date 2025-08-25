#!/bin/bash
set -e

echo "[Gdata Benchmark Suite] Auto setup starting..."

# System update
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git build-essential

# Create virtual env
python3 -m venv venv
source venv/bin/activate

# Upgrade pip & install Python deps
pip install --upgrade pip
pip install -r requirements.txt

echo "[Gdata Benchmark Suite] Setup complete!"
echo "Activate environment with: source venv/bin/activate"
echo "Run benchmarks with: ./benchmark_all.sh"
