from peft import LoraConfig, TaskType
from transformers import TrainingArguments
from trl import SFTConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

training_arg = TrainingArguments(
    output_dir="LoRA/outputs"
)
training_args = SFTConfig(
    output_dir = "LoRA/outputs",
    num_train_epochs=1,
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)


