#!/usr/bin/env python3
# Source/Reference:
# - Hugging Face Transformers pipeline: https://huggingface.co/docs/transformers/index
# - Model: distilbert-base-uncased-finetuned-sst-2-english (widely used baseline)
# Notes:
# - Uses torch + CUDA. If your GPU supports FP16 well, set USE_FP16=1 to test faster path.

import os, time, torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def main():
    assert torch.cuda.is_available(), "No CUDA device. Install GPU PyTorch & driver."
    total = int(os.getenv("SAMPLES", "20000"))
    batch = int(os.getenv("BATCH", "64"))
    use_fp16 = os.getenv("USE_FP16", "1") == "1"
    print(f"=== NLP Inference (GPU) | samples={total}, batch={batch}, fp16={use_fp16} ===")

    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if use_fp16 else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=dtype)
    clf = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)

    texts = ["This product is awesome!"] * total

    # warm-up (2 batches)
    _ = clf(texts[:batch*2], batch_size=batch)

    torch.cuda.synchronize()
    t0 = time.time()
    _ = clf(texts, batch_size=batch)
    torch.cuda.synchronize()
    dt = time.time() - t0
    ips = total / dt
    print(f"Finished in {dt:.2f} s | throughput ~ {ips:.1f} samples/sec (GPU)")

if __name__ == "__main__":
    main()
