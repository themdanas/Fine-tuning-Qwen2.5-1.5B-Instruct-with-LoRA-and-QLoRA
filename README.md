# Fine-Tuning Qwen2.5-1.5B-Instruct with LoRA and QLoRA

This repository contains a modular and reproducible pipeline for parameter-efficient fine-tuning (PEFT) of **Qwen/Qwen2.5-1.5B-Instruct** using Low-Rank Adaptation (LoRA) and QLoRA with Hugging Face `transformers`, `peft`, and `trl`.

---

## Overview

Fine-tuning large language models from scratch or full parameter updates is computationally expensive. This project implements LoRA to freeze the base model parameters and inject trainable low-rank decomposition matrices into the multi-head attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`).

### Architecture

```text
Qwen2ForCausalLM
│
├── Embedding (Frozen)
├── Transformer Block 0..27
│   ├── Self-Attention
│   │   ├── q_proj + LoRA Adapter (Trainable)
│   │   ├── k_proj + LoRA Adapter (Trainable)
│   │   ├── v_proj + LoRA Adapter (Trainable)
│   │   └── o_proj + LoRA Adapter (Trainable)
│   └── MLP (Frozen)
│       ├── gate_proj (Frozen)
│       ├── up_proj   (Frozen)
│       └── down_proj (Frozen)
└── LM Head (Frozen)
```

---

## Repository Structure

```text
.
├── configs/
│   └── train_config.py      # LoRA and SFTTrainer hyperparameters
├── utils/
│   ├── dataset.py           # Dataset loading and mapping logic
│   ├── formatting.py        # Alpaca-to-chat-template formatter
│   ├── model.py             # Base model and tokenizer loader
│   └── trainer.py           # SFTTrainer instantiation utility
├── data/                    # Directory for local training data (optional)
├── outputs/                 # Saved adapter checkpoints and trainer outputs
├── compare_models.py        # Side-by-side generation: Base vs LoRA model
├── inference.py             # Evaluation/inference script using trained adapter
├── merge_adapter.py         # Merges LoRA weights back into the base model
├── sanity_check.py          # Quick pipeline & backward pass verification
├── test.py                  # Environment verification for core dependencies
├── train_lora.py            # LoRA training execution script
├── train_qlora.py           # 4-bit QLoRA training script template
└── requirements.txt         # Project dependencies
```

---

## Configuration & Hyperparameters

Configurations are defined in `configs/train_config.py`:

### LoRA Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `r` | `16` | Rank of low-rank update matrices |
| `lora_alpha` | `32` | Scaling factor for LoRA updates |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj"]` | Targeted attention projection layers |
| `lora_dropout` | `0.05` | Dropout probability for LoRA layers |
| `bias` | `"none"` | Trainable bias specification |
| `task_type` | `TaskType.CAUSAL_LM` | Causal Language Modeling task |

### Training Hyperparameters

| Parameter | Value |
| :--- | :--- |
| Optimizer / Trainer | `trl.SFTTrainer` |
| Learning Rate | `2e-4` |
| Epochs | `1` |
| Per-Device Batch Size | `2` |
| Gradient Accumulation Steps | `8` |
| Effective Batch Size | `16` |
| Logging Steps | `10` |
| Checkpoint Save Strategy | `epoch` |

---

## Dataset & Formatting

The default dataset configured in `utils/dataset.py` is `yahma/alpaca-cleaned`.

Raw instruction data is transformed into Qwen2.5 chat template format (`apply_chat_template`) via `utils/formatting.py`:

```json
[
  {"role": "user", "content": "<instruction>\n\n<input>"},
  {"role": "assistant", "content": "<output>"}
]
```

---

## Installation & Setup

### 1. Clone Repository & Create Virtual Environment

```bash
git clone https://github.com/themdanas/Fine-tuning-Qwen2.5-1.5B-Instruct-with-LoRA-and-QLoRA.git
cd Fine-tuning-Qwen2.5-1.5B-Instruct-with-LoRA-and-QLoRA

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on QLoRA & bitsandbytes**: `bitsandbytes` requires a CUDA-enabled GPU (Linux/Windows). On macOS (MPS) or CPU-only systems, use standard LoRA (`train_lora.py`).

### 3. Verify Environment

```bash
python test.py
```

---

## Workflow

### Step 1: Run Sanity Check

Validate tokenizer configuration, dataset loading, and a single backward pass before launching full training:

```bash
python sanity_check.py
```

Expected output:
```text
Backward Successful
```

### Step 2: Train with LoRA

Launch the fine-tuning run:

```bash
python train_lora.py
```

Checkpoints and adapter weights will be saved to the configured output path under `outputs/`.

### Step 3: Run Inference

Test the fine-tuned adapter against user prompts:

```bash
python inference.py
```

Update `ADAPTER_PATH` in `inference.py` to point to your specific checkpoint if needed (e.g., `outputs/checkpoint-63`).

### Step 4: Compare Base vs. Fine-Tuned Model

Run qualitative side-by-side comparisons using identical prompts and generation settings:

```bash
python compare_models.py
```

### Step 5: Merge Adapter into Base Model

To export a standalone model with merged weights for zero-overhead inference or deployment:

```bash
python merge_adapter.py
```

The consolidated weights will be saved into the specified destination folder (e.g., `merged_model/`).

---

## QLoRA Training

For memory-constrained GPUs (e.g., free-tier Google Colab T4), QLoRA loads the base model in 4-bit NormalFloat (`NF4`) precision:

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

Pass `quantization_config=bnb_config` to `AutoModelForCausalLM.from_pretrained` inside `train_qlora.py`.

---

## Hardware Requirements

- **LoRA (FP16 / BF16)**: Recommended 16 GB+ VRAM GPU, or Apple Silicon Mac (M1/M2/M3/M4 with 16 GB+ unified memory).
- **QLoRA (4-bit)**: Can run on GPUs with 6 GB–12 GB VRAM (CUDA required for `bitsandbytes`).

---

## License

This project is licensed under the Apache 2.0 License.