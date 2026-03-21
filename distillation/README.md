# LFM2.5 Knowledge Distillation on Mac M4 (24GB)

## Two-Phase Offline Distillation

Phase 1: Extract teacher logits to disk (teacher loaded alone)
Phase 2: Train student against saved logits (student loaded alone)

This approach never loads both models simultaneously.

## Usage

```bash
# Step 1: Install dependencies
pip install torch mlx mlx-lm transformers datasets tqdm

# Step 2: Extract teacher logits (uses ~8-10GB RAM)
python extract_teacher_logits.py \
    --teacher_model "mlx-community/Qwen2.5-14B-Instruct-4bit" \
    --dataset "HuggingFaceFW/fineweb-edu-score-2" \
    --output_dir "./teacher_logits" \
    --num_samples 50000 \
    --seq_length 512 \
    --top_k_logits 64

# Step 3: Distill into LFM2.5 (uses ~12-14GB RAM)
python distill_student.py \
    --student_model "../model_repo" \
    --logits_dir "./teacher_logits" \
    --output_dir "./distilled_lfm2" \
    --epochs 3 \
    --batch_size 1 \
    --gradient_accumulation 8 \
    --lr 2e-5 \
    --temperature 4.0 \
    --alpha 0.7
```
