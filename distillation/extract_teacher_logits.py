"""
Phase 1: Extract teacher logits to disk.

Loads the teacher model ALONE (no student in memory), runs inference on
training data, and saves the top-K logits per token to disk in shards.

Designed for Mac M4 24GB — uses MLX for efficient Apple Silicon inference.
Falls back to PyTorch + MPS if MLX is not available.

Memory usage: ~8-10GB for a 14B-INT4 teacher.
"""

import argparse
import json
import os
import struct
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract teacher logits for distillation")
    parser.add_argument("--teacher_model", type=str, required=True,
                        help="HuggingFace model ID or local path (e.g., mlx-community/Qwen2.5-14B-Instruct-4bit)")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu-score-2",
                        help="HuggingFace dataset to extract logits from")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="text")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--top_k_logits", type=int, default=64,
                        help="Save top-K logits per position (not full vocab — saves 1000x disk space)")
    parser.add_argument("--shard_size", type=int, default=1000,
                        help="Number of samples per shard file")
    parser.add_argument("--use_mlx", action="store_true", default=True,
                        help="Use MLX for Apple Silicon (default: True)")
    parser.add_argument("--no_mlx", action="store_true",
                        help="Force PyTorch instead of MLX")
    return parser.parse_args()


def load_teacher_mlx(model_id: str):
    """Load teacher using MLX (optimized for Apple Silicon)."""
    try:
        from mlx_lm import load
        model, tokenizer = load(model_id)
        return model, tokenizer, "mlx"
    except ImportError:
        print("MLX not available, falling back to PyTorch")
        return None, None, None


def load_teacher_pytorch(model_id: str):
    """Load teacher using PyTorch with MPS backend."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="mps",  # Apple Silicon GPU
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer, "pytorch"


def load_dataset_texts(dataset_name: str, split: str, text_field: str, num_samples: int):
    """Load and yield text samples from a HuggingFace dataset."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True)
    texts = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        text = sample.get(text_field, "")
        if len(text) > 50:  # skip very short samples
            texts.append(text)
    return texts


def extract_topk_logits_mlx(model, tokenizer, text: str, seq_length: int, top_k: int):
    """Extract top-K logits using MLX."""
    import mlx.core as mx

    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) < 10:
        return None, None, None

    tokens = tokens[:seq_length]
    input_ids = mx.array([tokens])

    logits = model(input_ids)  # [1, seq_len, vocab_size]
    logits = logits[0]  # [seq_len, vocab_size]

    # Get top-K logits and their indices
    top_k_indices = mx.argpartition(logits, kth=-top_k, axis=-1)[:, -top_k:]  # approximate top-k
    top_k_values = mx.take_along_axis(logits, top_k_indices, axis=-1)

    # Sort within top-k for consistency
    sort_idx = mx.argsort(top_k_values, axis=-1)[:, ::-1]
    top_k_values = mx.take_along_axis(top_k_values, sort_idx, axis=-1)
    top_k_indices = mx.take_along_axis(top_k_indices, sort_idx, axis=-1)

    return (
        np.array(tokens, dtype=np.int32),
        np.array(top_k_values, dtype=np.float16),
        np.array(top_k_indices, dtype=np.int32),
    )


def extract_topk_logits_pytorch(model, tokenizer, text: str, seq_length: int, top_k: int):
    """Extract top-K logits using PyTorch + MPS."""
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=seq_length, truncation=True)
    if len(tokens) < 10:
        return None, None, None

    input_ids = torch.tensor([tokens], device="mps")

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0].float()  # [seq_len, vocab_size]

    # Get top-K
    top_k_values, top_k_indices = torch.topk(logits, k=top_k, dim=-1, sorted=True)

    return (
        np.array(tokens, dtype=np.int32),
        top_k_values.cpu().numpy().astype(np.float16),
        top_k_indices.cpu().numpy().astype(np.int32),
    )


def save_shard(shard_data: list, shard_idx: int, output_dir: str):
    """
    Save a shard of teacher logits to disk.

    Format per shard (numpy compressed):
    - tokens_{i}: input token IDs [seq_len]
    - values_{i}: top-K logit values [seq_len, K] in float16
    - indices_{i}: top-K vocab indices [seq_len, K] in int32
    """
    shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.npz")

    save_dict = {}
    for i, (tokens, values, indices) in enumerate(shard_data):
        save_dict[f"tokens_{i}"] = tokens
        save_dict[f"values_{i}"] = values
        save_dict[f"indices_{i}"] = indices

    np.savez_compressed(shard_path, **save_dict)
    return shard_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Save extraction config for the student training script
    config = {
        "teacher_model": args.teacher_model,
        "seq_length": args.seq_length,
        "top_k_logits": args.top_k_logits,
        "shard_size": args.shard_size,
        "num_samples": args.num_samples,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load teacher model
    print(f"Loading teacher model: {args.teacher_model}")
    model, tokenizer, backend = None, None, None

    if not args.no_mlx:
        model, tokenizer, backend = load_teacher_mlx(args.teacher_model)

    if model is None:
        model, tokenizer, backend = load_teacher_pytorch(args.teacher_model)

    print(f"Using backend: {backend}")

    # Choose extraction function
    extract_fn = extract_topk_logits_mlx if backend == "mlx" else extract_topk_logits_pytorch

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    texts = load_dataset_texts(args.dataset, args.dataset_split, args.dataset_text_field, args.num_samples)
    print(f"Loaded {len(texts)} text samples")

    # Extract logits
    shard_data = []
    shard_idx = 0
    total_tokens = 0

    for i, text in enumerate(tqdm(texts, desc="Extracting teacher logits")):
        result = extract_fn(model, tokenizer, text, args.seq_length, args.top_k_logits)

        if result[0] is None:
            continue

        tokens, values, indices = result
        shard_data.append((tokens, values, indices))
        total_tokens += len(tokens)

        # Save shard when full
        if len(shard_data) >= args.shard_size:
            path = save_shard(shard_data, shard_idx, args.output_dir)
            print(f"Saved shard {shard_idx} → {path} ({len(shard_data)} samples)")
            shard_data = []
            shard_idx += 1

    # Save remaining
    if shard_data:
        path = save_shard(shard_data, shard_idx, args.output_dir)
        print(f"Saved shard {shard_idx} → {path} ({len(shard_data)} samples)")
        shard_idx += 1

    # Summary
    disk_usage_estimate = total_tokens * args.top_k_logits * (2 + 4) / (1024**3)  # float16 + int32
    print(f"\nDone!")
    print(f"  Total shards: {shard_idx}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Estimated disk usage: {disk_usage_estimate:.1f} GB")
    print(f"  Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
