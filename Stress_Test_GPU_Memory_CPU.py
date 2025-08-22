#!/usr/bin/env python3
# Source/Reference:
# - Pattern mirrors community "memory pressure" tests but on CPU (NumPy) for baseline contrast
# - This does NOT prove GPU VRAM, only shows CPU RAM pressure & copy bandwidth feel.

import os, time, psutil
import numpy as np

def main():
    gb = int(os.getenv("ALLOC_GB", "16"))   # tổng dung lượng muốn alloca (GB)
    chunk_mb = int(os.getenv("CHUNK_MB", "256"))
    chunks = (gb * 1024) // chunk_mb
    print(f"=== CPU Memory Pressure: target={gb} GB, chunk={chunk_mb} MB ===")

    arrs = []
    created = 0
    t0 = time.time()
    for i in range(chunks):
        try:
            n_elem = (chunk_mb * 1024 * 1024) // 4  # float32 ~ 4B
            arrs.append(np.zeros(n_elem, dtype=np.float32))
            created += 1
            if (i+1) % 10 == 0:
                mem = psutil.virtual_memory()
                print(f"Chunks: {i+1}/{chunks} | RAM used: {(mem.total-mem.available)/1e9:.1f} GB")
        except MemoryError:
            print("MemoryError encountered (expected on big targets).")
            break

    dt = time.time() - t0
    realized_gb = created * chunk_mb / 1024
    print(f"Allocated ~{realized_gb:.1f} GB in {dt:.1f}s on CPU")

if __name__ == "__main__":
    main()
