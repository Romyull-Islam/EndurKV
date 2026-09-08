"""Per-model KV cache architecture parameters.

These are the fp16 KV cache footprints used by llama.cpp's default cache type
(GGML_TYPE_F16). Quantized weights (Q4_K_M etc.) do NOT change KV cache size
because the cache itself is fp16; only model weights are quantized.

Source: model config.json / GGUF metadata. Verified by counting layers in the
xarch ATNH captures (n_layers field matches).
"""

# Per-model fp16 KV-cache parameters.
# Bytes-per-token = n_layers * n_head_kv * 2 (K and V) * head_dim * 2 (fp16 bytes)
MODEL_SPECS = {
    "llama3.2-1b": {
        "n_layers": 16, "n_head_kv": 8, "head_dim": 64,
        "n_head_q": 32, "attn_kind": "GQA-4",
        "full_name": "Llama-3.2-1B-Instruct",
    },
    "llama3.2-3b": {
        "n_layers": 28, "n_head_kv": 8, "head_dim": 128,
        "n_head_q": 24, "attn_kind": "GQA-3",
        "full_name": "Llama-3.2-3B-Instruct",
    },
    "llama3.1-8b": {
        "n_layers": 32, "n_head_kv": 8, "head_dim": 128,
        "n_head_q": 32, "attn_kind": "GQA-4",
        "full_name": "Llama-3.1-8B-Instruct",
    },
    "mistral-7b": {
        "n_layers": 32, "n_head_kv": 8, "head_dim": 128,
        "n_head_q": 32, "attn_kind": "GQA-4",
        "full_name": "Mistral-7B-Instruct-v0.3",
    },
    "phi3-mini-4k": {
        "n_layers": 32, "n_head_kv": 32, "head_dim": 96,
        "n_head_q": 32, "attn_kind": "MHA",
        "full_name": "Phi-3-mini-4k-instruct",
    },
    "qwen2-7b": {
        "n_layers": 28, "n_head_kv": 4, "head_dim": 128,
        "n_head_q": 28, "attn_kind": "GQA-7",
        "full_name": "Qwen2-7B-Instruct",
    },
    "gemma2-2b": {
        # Gemma-2 uses hybrid global+sliding attention. We report the GLOBAL
        # layer footprint here (sliding layers store a small fixed window).
        "n_layers": 26, "n_head_kv": 4, "head_dim": 256,
        "n_head_q": 8, "attn_kind": "GQA-2 (hybrid)",
        "full_name": "Gemma-2-2B-it",
    },
    "r1distill-llama-8b": {
        "n_layers": 32, "n_head_kv": 8, "head_dim": 128,
        "n_head_q": 32, "attn_kind": "GQA-4",
        "full_name": "DeepSeek-R1-Distill-Llama-8B",
    },
}


def kv_bytes_per_token(model_key: str, dtype_bytes: int = 2) -> int:
    """Per-token KV cache footprint in bytes for a given model.

    dtype_bytes = 2 for fp16 (llama.cpp default), 1 for int8, etc.
    """
    s = MODEL_SPECS[model_key]
    return s["n_layers"] * s["n_head_kv"] * 2 * s["head_dim"] * dtype_bytes


def kv_memory_mb(model_key: str, n_kv_positions: int, dtype_bytes: int = 2) -> float:
    """Memory occupancy in MB for a given cache fill."""
    return kv_bytes_per_token(model_key, dtype_bytes) * n_kv_positions / (1024 * 1024)


def kv_memory_per_head_mb(model_key: str, kept_per_head: float,
                           dtype_bytes: int = 2) -> float:
    """Memory after per-head eviction (when policy keeps `kept_per_head`
    positions per head per layer).
    """
    s = MODEL_SPECS[model_key]
    return (s["n_layers"] * s["n_head_kv"] * 2 * s["head_dim"]
            * dtype_bytes * kept_per_head) / (1024 * 1024)


def infer_model_from_dir(dir_name: str) -> str:
    """Best-effort model-key inference from a logs/ dir name."""
    d = dir_name.lower()
    # Order matters — check specific names before generic ones
    if "r1distill" in d or "r1-distill" in d:
        return "r1distill-llama-8b"
    if "1b" in d:
        return "llama3.2-1b"
    if "3b" in d:
        return "llama3.2-3b"
    if "llama8b" in d or "_8b_" in d or d.endswith("_8b"):
        return "llama3.1-8b"
    if "mistral" in d:
        return "mistral-7b"
    if "phi3" in d:
        return "phi3-mini-4k"
    if "qwen2" in d or "qwen-2" in d:
        return "qwen2-7b"
    if "gemma2" in d or "gemma-2" in d:
        return "gemma2-2b"
    return ""


if __name__ == "__main__":
    print("Model key            | bytes/token | 4K cache | 8K cache | 13K cache")
    print("-" * 72)
    for key in MODEL_SPECS:
        bpt = kv_bytes_per_token(key)
        print(f"{key:20} | {bpt:>11,} | {bpt*4096/1e6:>7.1f}M | "
              f"{bpt*8192/1e6:>7.1f}M | {bpt*13312/1e6:>7.1f}M")
