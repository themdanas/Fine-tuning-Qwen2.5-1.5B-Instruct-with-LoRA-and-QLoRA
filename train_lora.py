from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


from utils.dataset import load_training_dataset
from utils.model import load_model_and_tokenizer, MODEL_NAME
from configs.train_config import lora_config, training_args
from utils.trainer import get_trainer

model, tokenizer = load_model_and_tokenizer()

dataset = load_training_dataset(tokenizer)

trainer = get_trainer(
    model=model,
    dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()