import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "LoRA/outputs/checkpoint-63"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

#base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

lora_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    ),
    ADAPTER_PATH,
)

base_model.eval()
lora_model.eval()

def generate(model, prompt):

    messages = [
        {
            "role":"user",
            "content":prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensor="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temprature=0.7,
            do_sample=True,
        )

    generated = output[0][input["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

prompt = "Give three tips for staying healthy."

print("=" * 80)
print("BASE MODEL")
print("=" * 80)

print(generate(base_model, prompt))

print()

print("=" * 80)
print("LORA MODEL")
print("=" * 80)

print(generate(lora_model, prompt))