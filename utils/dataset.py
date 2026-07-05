from datasets import load_dataset
from utils.formatting import formatting_prompts_func


def load_training_dataset(tokenizer):
    dataset = load_dataset(
        "yahma/alpaca-cleaned",
        split="train",
    )

    dataset = dataset.map(formatting_prompts_func, 
    fn_kwargs={"tokenizer":tokenizer})

    return dataset
