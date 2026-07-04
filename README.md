# Fine-tuning-Qwen2.5-1.5B-Instruct-with-LoRA-and-QLoRA
Qwen2ForCausalLM
│
├── Embedding (Frozen)
├── Transformer Block
│      │
│      ├── q_proj + LoRA
│      ├── k_proj + LoRA
│      ├── v_proj + LoRA
│      └── o_proj + LoRA
│
├── Transformer Block
│      │
│      ├── q_proj + LoRA
│      ├── ...
│
└── LM Head (Frozen)