#!/bin/bash
# =============================================================================
# Full distillation pipeline for LFM2.5 on Mac M4 (24GB RAM)
#
# Two-phase approach:
#   Phase 1: Load teacher ALONE → extract logits to disk
#   Phase 2: Load student ALONE → train against saved logits
#
# Total time estimate:
#   Phase 1 (50K samples, 14B teacher): ~4-8 hours
#   Phase 2 (3 epochs):                 ~6-12 hours
#   Total:                              ~10-20 hours
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Configuration ----

# Teacher model options (pick ONE, comment out the rest):
#
# Option A: Best quality, ~8GB RAM, slower extraction
TEACHER="mlx-community/Qwen2.5-14B-Instruct-4bit"
#
# Option B: Good quality, ~4GB RAM, faster extraction
# TEACHER="mlx-community/Qwen2.5-7B-Instruct-4bit"
#
# Option C: Decent quality, ~2.5GB RAM, fastest extraction
# TEACHER="mlx-community/Qwen2.5-3B-Instruct-4bit"

STUDENT="../model_repo"
LOGITS_DIR="./teacher_logits"
OUTPUT_DIR="./distilled_lfm2"

NUM_SAMPLES=50000       # More = better but slower. 50K is a good start.
SEQ_LENGTH=512          # Max 1024 before memory gets tight
TOP_K=64                # Top-64 logits per position (covers >99% probability mass)

# ---- Phase 1: Extract teacher logits ----

echo "============================================="
echo "Phase 1: Extracting teacher logits"
echo "Teacher: $TEACHER"
echo "Samples: $NUM_SAMPLES"
echo "============================================="

python extract_teacher_logits.py \
    --teacher_model "$TEACHER" \
    --dataset "HuggingFaceFW/fineweb-edu-score-2" \
    --output_dir "$LOGITS_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --seq_length "$SEQ_LENGTH" \
    --top_k_logits "$TOP_K" \
    --shard_size 1000

echo ""
echo "Phase 1 complete. Logits saved to $LOGITS_DIR"
echo ""

# ---- Phase 2: Distill into student ----

echo "============================================="
echo "Phase 2: Distilling into LFM2.5"
echo "Student: $STUDENT"
echo "============================================="

python distill_student.py \
    --student_model "$STUDENT" \
    --logits_dir "$LOGITS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 3 \
    --batch_size 1 \
    --gradient_accumulation 8 \
    --lr 2e-5 \
    --temperature 4.0 \
    --alpha 0.7 \
    --save_every 500 \
    --log_every 10 \
    --device mps

echo ""
echo "============================================="
echo "Distillation complete!"
echo "Final model: $OUTPUT_DIR/final/"
echo "============================================="
