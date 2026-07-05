from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model
from configs.train_config import lora_config

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    return model,tokenizer


