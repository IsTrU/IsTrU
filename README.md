# LiquidAI/LFM2.5-1.2B-Base - Architecture Exploration

A deep dive into the architecture and codebase of Liquid AI's LFM2.5-1.2B-Base language model.

## Model Overview

| Property | Value |
|---|---|
| **Provider** | Liquid AI |
| **Model Type** | `lfm2` |
| **Architecture** | `Lfm2ForCausalLM` |
| **Parameters** | ~1.17 billion |
| **Hidden Size** | 2048 |
| **Layers** | 16 (10 conv + 6 attention) |
| **Context Length** | 32,768 tokens (max 128K positional embeddings) |
| **Vocab Size** | 65,536 |
| **Training Data** | 28 trillion tokens |
| **Precision** | bfloat16 |
| **Languages** | English, Arabic, Chinese, French, German, Japanese, Korean, Spanish |

## Architecture Overview

LFM2.5 is a **hybrid convolution-attention** model. Unlike pure transformers, it interleaves lightweight depthwise convolution layers ("short conv" or LIV blocks) with standard grouped query attention (GQA) layers. This design is optimized for efficient on-device inference -- convolution layers are much cheaper than attention at inference time.

### Layer Layout (16 layers)

```
Layer  0: Conv          (Lfm2ShortConv)
Layer  1: Conv          (Lfm2ShortConv)
Layer  2: Attention     (Lfm2Attention + GQA)
Layer  3: Conv          (Lfm2ShortConv)
Layer  4: Conv          (Lfm2ShortConv)
Layer  5: Attention     (Lfm2Attention + GQA)
Layer  6: Conv          (Lfm2ShortConv)
Layer  7: Conv          (Lfm2ShortConv)
Layer  8: Attention     (Lfm2Attention + GQA)
Layer  9: Conv          (Lfm2ShortConv)
Layer 10: Attention     (Lfm2Attention + GQA)
Layer 11: Conv          (Lfm2ShortConv)
Layer 12: Attention     (Lfm2Attention + GQA)
Layer 13: Conv          (Lfm2ShortConv)
Layer 14: Attention     (Lfm2Attention + GQA)
Layer 15: Conv          (Lfm2ShortConv)
```

Pattern: The early layers favor consecutive conv blocks (conv-conv-attn), while the later layers alternate more frequently (conv-attn-conv-attn-conv).

### Inheritance Chain

The modular source (`modular_lfm2.py`) reveals the inheritance:

```
Lfm2RMSNorm          -> LlamaRMSNorm
Lfm2RotaryEmbedding  -> Gemma2RotaryEmbedding
Lfm2Attention        -> LlamaAttention (with QK LayerNorm + renamed projections)
Lfm2PreTrainedModel  -> LlamaPreTrainedModel
Lfm2Model            -> LlamaModel (with custom forward + embedding_norm)
Lfm2ForCausalLM      -> LlamaForCausalLM (pass-through, no changes)
Lfm2ShortConv        -> nn.Module (fully custom)
Lfm2DecoderLayer     -> GradientCheckpointingLayer (fully custom)
Lfm2HybridConvCache  -> custom (not inheriting from Cache)
Lfm2MLP              -> nn.Module (custom SwiGLU, not inheriting from LlamaMLP)
```

## Core Components

### 1. Lfm2ShortConv (The LIV Convolution Block)

This is the novel component. Each conv layer uses a **double-gated depthwise convolution**:

```
Input x [batch, seq, 2048]
    |
    v
in_proj: Linear(2048 -> 6144)     # Projects to 3x hidden_size
    |
    v
Split into B, C, x               # Three chunks of [batch, 2048, seq]
    |
    v
Bx = B * x                       # First gate: element-wise multiply
    |
    v
conv1d(Bx)                       # Depthwise causal Conv1d (kernel=3, groups=2048)
    |
    v
y = C * conv_out                 # Second gate: element-wise multiply
    |
    v
out_proj: Linear(2048 -> 2048)   # Project back to hidden size
```

Key details:
- **Depthwise convolution**: `groups=hidden_size` (2048), so each channel has its own kernel
- **Kernel size**: 3 (`conv_L_cache = 3`) -- very short, captures local patterns
- **Causal**: padding ensures no future information leakage
- **Double gating**: input is projected to 3x, split into B (gate 1), C (gate 2), and x (data). `B*x` is convolved, then gated by `C`
- **No bias** by default (`conv_bias = false`)
- **Fast path**: uses `causal_conv1d` CUDA kernels when available, otherwise falls back to PyTorch

### 2. Lfm2Attention (Grouped Query Attention)

Standard GQA with two additions: **QK LayerNorm** and renamed output projection.

```
Input [batch, seq, 2048]
    |
    +---> q_proj(2048 -> 2048) -> q_layernorm -> reshape [batch, 32, seq, 64]
    +---> k_proj(2048 -> 512)  -> k_layernorm -> reshape [batch, 8, seq, 64]
    +---> v_proj(2048 -> 512)  ->              -> reshape [batch, 8, seq, 64]
    |
    v
RoPE applied to Q and K (theta=1,000,000)
    |
    v
Attention: Q @ K^T / sqrt(64) -> softmax -> @ V
    |
    v
out_proj: Linear(2048 -> 2048)
```

| Parameter | Value |
|---|---|
| Attention heads | 32 |
| KV heads | 8 (4:1 GQA ratio) |
| Head dimension | 64 (2048 / 32) |
| RoPE theta | 1,000,000 |
| QK LayerNorm | RMSNorm per head |
| Bias | None |

Supports Flash Attention 2, SDPA, and Flex Attention backends.

### 3. Lfm2DecoderLayer (Hybrid Layer)

Each layer conditionally uses either attention or convolution:

```python
if self.is_attention_layer:
    hidden = self.self_attn(operator_norm(hidden), ...)  # Attention
else:
    hidden = self.conv(operator_norm(hidden), ...)       # ShortConv

hidden = hidden + residual                               # Residual connection
hidden = hidden + self.feed_forward(ffn_norm(hidden))    # FFN with residual
```

Both layer types share the same MLP (feed-forward) block and use pre-norm (RMSNorm before operator).

### 4. Lfm2MLP (SwiGLU Feed-Forward)

```
Input x [batch, seq, 2048]
    |
    +---> w1: Linear(2048 -> 8192) -> SiLU activation
    +---> w3: Linear(2048 -> 8192)  (gate)
    |
    v
    SiLU(w1(x)) * w3(x)           # Gated activation
    |
    v
    w2: Linear(8192 -> 2048)       # Project back
```

The intermediate size is auto-adjusted: `int(2/3 * 12288) = 8192`, rounded to multiple of 256.

### 5. Lfm2HybridConvCache

A custom cache that manages **both** attention KV cache and conv rolling state:
- **Attention layers**: standard KV cache `[batch, heads, seq, head_dim]`
- **Conv layers**: rolling buffer `[batch, hidden_size, L_cache]` (shape `[batch, 2048, 3]`)

This unified cache enables efficient autoregressive generation with both layer types.

### 6. Lfm2Model (Top-Level)

```
Input token IDs
    |
    v
embed_tokens: Embedding(65536, 2048)    # Token embeddings (tied with lm_head)
    |
    v
rotary_emb: RoPE (computed once, shared)
    |
    v
16x Lfm2DecoderLayer                    # Hybrid layers
    |                                    # Conv layers get linear (2D) mask
    |                                    # Attn layers get full causal mask
    v
embedding_norm: RMSNorm(2048)           # Final norm (named embedding_norm, not norm)
    |
    v
lm_head: Linear(2048 -> 65536)          # Output logits (weights tied with embed_tokens)
```

Notable: Conv layers receive the raw 2D attention mask (for padding), while attention layers receive the full 4D causal mask.

## Configuration Parameters

From `config.json`:

```json
{
  "model_type": "lfm2",
  "architectures": ["Lfm2ForCausalLM"],
  "hidden_size": 2048,
  "intermediate_size": 12288,
  "num_hidden_layers": 16,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 65536,
  "max_position_embeddings": 128000,
  "rope_theta": 1000000.0,
  "conv_L_cache": 3,
  "conv_bias": false,
  "block_auto_adjust_ff_dim": true,
  "block_use_swiglu": true,
  "norm_eps": 1e-05,
  "tie_embedding": true,
  "dtype": "bfloat16",
  "bos_token_id": 1,
  "eos_token_id": 7,
  "pad_token_id": 0
}
```

## Tokenizer

| Property | Value |
|---|---|
| Vocab size | 65,536 |
| BOS token | `<\|startoftext\|>` (id: 1) |
| EOS token | `<\|im_end\|>` (id: 7) |
| PAD token | `<\|pad\|>` (id: 0) |
| Chat template | ChatML-style (`<\|im_start\|>role ... <\|im_end\|>`) |
| Tool support | Yes (via `<\|tool_list_start\|>` / `<\|tool_response_start\|>` tags) |

## File Structure

```
.
├── README.md                          # This file
├── model_repo/                        # Downloaded from HuggingFace
│   ├── config.json                    # Model architecture config
│   ├── generation_config.json         # Generation defaults
│   ├── tokenizer.json                 # Full tokenizer vocabulary
│   ├── tokenizer_config.json          # Tokenizer settings
│   ├── special_tokens_map.json        # Special token definitions
│   ├── chat_template.jinja            # ChatML chat template
│   ├── LICENSE                        # License
│   └── README.md                      # Original model card
└── architecture_source/               # Extracted from transformers v5.3.0
    ├── __init__.py                    # Module exports
    ├── configuration_lfm2.py          # Lfm2Config class
    ├── modeling_lfm2.py               # Full modeling code (auto-generated from modular)
    └── modular_lfm2.py               # Modular source showing inheritance chain
```

Note: `model.safetensors` (2.34 GB weights) was intentionally not downloaded -- only config and code files are needed for architecture analysis.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2.5-1.2B-Base",
    device_map="auto",
    torch_dtype="bfloat16",
)
tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-1.2B-Base")

inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.2, top_p=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Key Design Decisions

1. **Hybrid over pure transformer**: 10 conv + 6 attn layers means ~63% of layers avoid quadratic attention cost, enabling efficient long-context inference on edge devices.

2. **Double-gated short convolution**: The LIV block uses two multiplicative gates (B and C) around a tiny kernel-3 depthwise conv. This is much cheaper than attention but can still capture local sequential patterns.

3. **QK LayerNorm**: RMSNorm applied per-head to queries and keys before RoPE, stabilizing training at scale.

4. **Tied embeddings**: Input embedding and output lm_head share weights, reducing parameter count.

5. **SwiGLU MLP**: Using gated activation (SiLU * gate) with auto-adjusted intermediate size (8192 effective).

6. **Very high RoPE theta** (1M): Enables extrapolation to long sequences beyond the 32K training context.

## References

- [HuggingFace Model Card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base)
- [LFM2 Technical Report (arXiv:2511.23404)](https://arxiv.org/abs/2511.23404)
- [Introducing LFM2.5 Blog Post](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
- [Liquid AI Documentation](https://docs.liquid.ai/lfm)
