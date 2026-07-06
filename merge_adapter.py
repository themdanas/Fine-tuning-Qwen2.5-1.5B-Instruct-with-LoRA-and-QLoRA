from dill.logger import adapter
from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER = "LoRA/outputs/checkpoint-63"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    model,
    ADAPTER,
)

print("Merging...")

merged_model = model.merge_and_unload()

merged_model.save_pretrained("merged model")

print("Done!")