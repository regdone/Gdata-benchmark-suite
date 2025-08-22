#!/usr/bin/env python3
# Source/Reference:
# - Hugging Face Transformers pipeline: https://huggingface.co/docs/transformers/index
# - Model: distilbert-base-uncased-finetuned-sst-2-english (very popular sentiment classifier)

import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from transformers import pipeline

def main():
    total = int(os.getenv("SAMPLES", "10000"))
    batch = int(os.getenv("BATCH", "32"))
    print(f"=== NLP Inference (CPU) | samples={total}, batch={batch} ===")

    clf = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

    texts = ["This product is awesome!"] * total

    # warm-up
    _ = clf(texts[:128], batch_size=batch)

    t0 = time.time()
    _ = clf(texts, batch_size=batch)
    dt = time.time() - t0
    ips = total / dt
    print(f"Finished in {dt:.2f} s | throughput ~ {ips:.1f} samples/sec (CPU)")

if __name__ == "__main__":
    main()
