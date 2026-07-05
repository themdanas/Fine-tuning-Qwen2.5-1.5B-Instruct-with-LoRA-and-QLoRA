from utils.model import load_model_and_tokenizer
from utils.dataset import load_training_dataset
import torch
model, tokenizer = load_model_and_tokenizer()

dataset = load_training_dataset(tokenizer)

sample = dataset[0]["text"]

inputs = tokenizer(
    sample,
    return_tensors="pt"
)

outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    labels=inputs["input_ids"],
)

outputs.loss.backward()

print("Backward Successful")