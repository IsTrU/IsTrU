"""
Phase 2: Distill teacher knowledge into LFM2.5 student.

Loads the student model ALONE (teacher logits come from disk).
Uses the top-K sparse logits saved by extract_teacher_logits.py
to compute KL divergence loss without needing the full teacher vocab.

Designed for Mac M4 24GB:
- Student (1.2B BF16):         ~2.4 GB
- Optimizer (AdamW):           ~4.8 GB
- Gradients:                   ~2.4 GB
- Activations (bs=1, seq=512): ~1.0 GB
- Total:                       ~10.6 GB (well within 24GB)

Memory usage: ~11-14GB depending on sequence length.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Distill teacher into LFM2.5")
    parser.add_argument("--student_model", type=str, required=True,
                        help="Path to LFM2.5 model directory (e.g., ../model_repo)")
    parser.add_argument("--logits_dir", type=str, required=True,
                        help="Path to extracted teacher logits directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save the distilled model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Micro batch size (keep at 1 for 24GB)")
    parser.add_argument("--gradient_accumulation", type=int, default=8,
                        help="Effective batch = batch_size × gradient_accumulation")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min_lr", type=float, default=2e-6)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Distillation temperature (higher = softer distributions)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight for KD loss (1-alpha = weight for CE loss)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="mps",
                        help="Device: mps (Apple Silicon), cpu, or cuda")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset: reads pre-computed teacher logits from disk shards
# ---------------------------------------------------------------------------

class TeacherLogitsDataset(Dataset):
    """
    Lazy-loads teacher logits from disk shards.
    Each sample: (input_ids, teacher_top_k_values, teacher_top_k_indices)
    """

    def __init__(self, logits_dir: str, student_vocab_size: int):
        self.logits_dir = logits_dir
        self.student_vocab_size = student_vocab_size

        # Load extraction config
        with open(os.path.join(logits_dir, "config.json")) as f:
            self.config = json.load(f)
        self.top_k = self.config["top_k_logits"]

        # Index all shards and samples
        self.shard_files = sorted(Path(logits_dir).glob("shard_*.npz"))
        self.index = []  # list of (shard_idx, sample_idx)

        for shard_idx, shard_file in enumerate(self.shard_files):
            # Peek at shard to count samples
            data = np.load(shard_file)
            sample_idx = 0
            while f"tokens_{sample_idx}" in data:
                self.index.append((shard_idx, sample_idx))
                sample_idx += 1
            data.close()

        print(f"Loaded {len(self.index)} samples from {len(self.shard_files)} shards")

        # Cache the currently loaded shard to avoid re-reading
        self._cached_shard_idx = -1
        self._cached_shard_data = None

    def __len__(self):
        return len(self.index)

    def _load_shard(self, shard_idx: int):
        if shard_idx != self._cached_shard_idx:
            self._cached_shard_data = np.load(self.shard_files[shard_idx])
            self._cached_shard_idx = shard_idx

    def __getitem__(self, idx):
        shard_idx, sample_idx = self.index[idx]
        self._load_shard(shard_idx)

        data = self._cached_shard_data
        tokens = data[f"tokens_{sample_idx}"]        # [seq_len]
        values = data[f"values_{sample_idx}"]         # [seq_len, K] float16
        indices = data[f"indices_{sample_idx}"]       # [seq_len, K] int32

        # Filter out teacher vocab indices that exceed student vocab
        mask = indices < self.student_vocab_size
        # Zero out entries that reference tokens outside student vocab
        indices = np.where(mask, indices, 0)
        values = np.where(mask, values, -1e9)  # -inf for out-of-vocab

        return {
            "input_ids": torch.from_numpy(tokens.astype(np.int64)),
            "teacher_values": torch.from_numpy(values.astype(np.float32)),
            "teacher_indices": torch.from_numpy(indices.astype(np.int64)),
        }


def collate_fn(batch):
    """Pad sequences in batch to same length."""
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    top_k = batch[0]["teacher_values"].shape[-1]
    teacher_values = torch.full((len(batch), max_len, top_k), -1e9)
    teacher_indices = torch.zeros(len(batch), max_len, top_k, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].shape[0]
        input_ids[i, :seq_len] = b["input_ids"]
        teacher_values[i, :seq_len] = b["teacher_values"]
        teacher_indices[i, :seq_len] = b["teacher_indices"]
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "teacher_values": teacher_values,
        "teacher_indices": teacher_indices,
        "attention_mask": attention_mask,
    }


# ---------------------------------------------------------------------------
# Distillation loss: sparse KL divergence from top-K teacher logits
# ---------------------------------------------------------------------------

def sparse_kd_loss(
    student_logits: torch.Tensor,     # [batch, seq_len, vocab_size]
    teacher_values: torch.Tensor,     # [batch, seq_len, K]
    teacher_indices: torch.Tensor,    # [batch, seq_len, K]
    temperature: float,
    attention_mask: torch.Tensor,     # [batch, seq_len]
) -> torch.Tensor:
    """
    Compute KL divergence between student and teacher using ONLY the
    teacher's top-K logits. This is a sparse approximation that:
    1. Avoids needing the full teacher vocab in memory
    2. Focuses learning on the tokens the teacher considers most likely
    3. Is mathematically close to full KD (top-64 covers >99% of probability mass)

    KL(teacher || student) = sum_k teacher_prob_k * log(teacher_prob_k / student_prob_k)
    """
    batch_size, seq_len, vocab_size = student_logits.shape

    # Apply temperature scaling
    teacher_scaled = teacher_values / temperature  # [B, S, K]
    teacher_probs = F.softmax(teacher_scaled, dim=-1)  # [B, S, K]

    # Gather student logits at teacher's top-K positions
    student_at_teacher = torch.gather(student_logits, dim=-1, index=teacher_indices)  # [B, S, K]
    student_scaled = student_at_teacher / temperature
    student_log_probs = F.log_softmax(student_scaled, dim=-1)  # [B, S, K]

    # KL divergence per position
    kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)  # [B, S]

    # Mask out padding — shift by 1 since we predict next token
    # For position i, student_logits[i] predicts token at position i+1
    mask = attention_mask[:, 1:].float()  # [B, S-1]
    kl_per_token = kl_per_token[:, :-1]   # [B, S-1] (align with next-token prediction)

    # Mean over non-padding tokens
    loss = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)

    # Scale by T² (standard distillation scaling — Hinton et al. 2015)
    loss = loss * (temperature ** 2)

    return loss


def ce_loss(
    student_logits: torch.Tensor,   # [B, S, V]
    input_ids: torch.Tensor,        # [B, S]
    attention_mask: torch.Tensor,   # [B, S]
) -> torch.Tensor:
    """Standard cross-entropy on ground truth tokens (next-token prediction)."""
    # Shift: predict position i+1 from position i
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].float()

    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)

    loss = (loss_per_token * shift_mask).sum() / shift_mask.sum().clamp(min=1)
    return loss


# ---------------------------------------------------------------------------
# Learning rate scheduler: cosine with warmup
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load student model
    # -----------------------------------------------------------------------
    print(f"Loading student model from: {args.student_model}")

    # Add architecture source to path so transformers can find the model class
    import sys
    arch_source = os.path.join(os.path.dirname(args.student_model), "architecture_source")
    if os.path.exists(arch_source):
        sys.path.insert(0, arch_source)

    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(args.student_model, trust_remote_code=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    student = student.to(device)
    student.train()

    vocab_size = config.vocab_size
    print(f"Student loaded: {sum(p.numel() for p in student.parameters()):,} parameters, vocab={vocab_size}")

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    dataset = TeacherLogitsDataset(args.logits_dir, student_vocab_size=vocab_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # keep memory low on Mac
        pin_memory=False,
    )

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=False,  # fused not supported on MPS
    )

    total_steps = (len(dataloader) * args.epochs) // args.gradient_accumulation
    print(f"Total optimization steps: {total_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"Temperature: {args.temperature}, Alpha (KD weight): {args.alpha}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    global_step = 0
    best_loss = float("inf")
    log_losses = {"total": 0, "kd": 0, "ce": 0, "count": 0}

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            teacher_values = batch["teacher_values"].to(device)
            teacher_indices = batch["teacher_indices"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            student_logits = outputs.logits  # [B, S, V]

            # Compute losses
            loss_kd = sparse_kd_loss(
                student_logits=student_logits,
                teacher_values=teacher_values,
                teacher_indices=teacher_indices,
                temperature=args.temperature,
                attention_mask=attention_mask,
            )

            loss_ce = ce_loss(
                student_logits=student_logits,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Combined loss: alpha * KD + (1 - alpha) * CE
            loss = args.alpha * loss_kd + (1 - args.alpha) * loss_ce
            loss = loss / args.gradient_accumulation

            # Backward pass
            loss.backward()

            # Logging
            log_losses["total"] += loss.item() * args.gradient_accumulation
            log_losses["kd"] += loss_kd.item()
            log_losses["ce"] += loss_ce.item()
            log_losses["count"] += 1

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)

                # LR schedule
                lr = get_lr(global_step, args.warmup_steps, total_steps, args.lr, args.min_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                if global_step % args.log_every == 0:
                    avg_total = log_losses["total"] / log_losses["count"]
                    avg_kd = log_losses["kd"] / log_losses["count"]
                    avg_ce = log_losses["ce"] / log_losses["count"]
                    print(
                        f"\n  Step {global_step}/{total_steps} | "
                        f"LR: {lr:.2e} | "
                        f"Loss: {avg_total:.4f} (KD: {avg_kd:.4f}, CE: {avg_ce:.4f})"
                    )
                    log_losses = {"total": 0, "kd": 0, "ce": 0, "count": 0}

                # Save checkpoint
                if global_step % args.save_every == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    student.save_pretrained(ckpt_dir)
                    print(f"\n  Saved checkpoint → {ckpt_dir}")

            # Free memory aggressively on constrained device
            del outputs, student_logits, loss, loss_kd, loss_ce
            if device.type == "mps":
                torch.mps.empty_cache()

    # -----------------------------------------------------------------------
    # Save final model
    # -----------------------------------------------------------------------
    final_dir = os.path.join(args.output_dir, "final")
    student.save_pretrained(final_dir)
    print(f"\nDistillation complete! Final model saved to: {final_dir}")

    # Save distillation config for reproducibility
    distill_config = {
        "teacher_model": json.load(open(os.path.join(args.logits_dir, "config.json")))["teacher_model"],
        "student_model": args.student_model,
        "epochs": args.epochs,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "lr": args.lr,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "total_steps": total_steps,
        "global_steps_completed": global_step,
    }
    with open(os.path.join(final_dir, "distillation_config.json"), "w") as f:
        json.dump(distill_config, f, indent=2)


if __name__ == "__main__":
    main()
