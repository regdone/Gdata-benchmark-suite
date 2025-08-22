#!/usr/bin/env python3
# Source/Reference:
# - Community-style CUDA VRAM stress patterns frequently used to find max batch size
# - Uses PyTorch CUDA tensors; increments by CHUNK_MB until OOM, then reports peak

import os, time, torch

def main():
    assert torch.cuda.is_available(), "No CUDA device. Check driver/CUDA/PyTorch."
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)

    chunk_mb = int(os.getenv("CHUNK_MB", "512"))
    max_gb_hint = float(os.getenv("MAX_GB_HINT", "80"))  # stop early if needed

    print(f"=== GPU VRAM Stress (PyTorch) | chunk={chunk_mb} MB ===")
    tensors = []
    created = 0
    t0 = time.time()
    try:
        while True:
            # Allocate a CHUNK_MB float16 tensor (to stress VRAM faster)
            num_elems = (chunk_mb * 1024 * 1024) // 2  # fp16 -> 2 bytes
            t = torch.zeros(num_elems, dtype=torch.float16, device=device)
            tensors.append(t)
            created += 1

            if created % 10 == 0:
                cur = torch.cuda.memory_allocated(device) / (1024**3)
                peak = torch.cuda.max_memory_allocated(device) / (1024**3)
                print(f"Chunks: {created} | allocated: {cur:.2f} GB | peak: {peak:.2f} GB")
            if (created * chunk_mb) / 1024.0 > max_gb_hint:
                print("Max GB hint reached, stopping early.")
                break
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("Hit CUDA OOM (expected).")
        else:
            raise

    dt = time.time() - t0
    cur = torch.cuda.memory_allocated(device) / (1024**3)
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(f"Final allocated: {cur:.2f} GB | peak: {peak:.2f} GB | time: {dt:.1f}s")

    # Optional simple bandwidth-ish op
    if len(tensors) >= 2:
        print("Testing a large vector add on GPU...")
        a, b = tensors[-1], tensors[-2]
        torch.cuda.synchronize()
        t1 = time.time()
        _ = a.add_(b)
        torch.cuda.synchronize()
        dt2 = time.time() - t1
        bytes_moved = (a.numel() * a.element_size()) * 2  # read a+b
        print(f"VecAdd time: {dt2:.4f}s ~ {(bytes_moved/dt2)/1e9:.1f} GB/s (very rough)")

if __name__ == "__main__":
    main()
