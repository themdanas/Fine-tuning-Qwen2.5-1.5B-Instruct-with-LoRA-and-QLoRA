from trl.data_utils import apply_chat_template


def formatting_prompts_func(example, tokenizer):
    if example["input"]:
        user_content = (
            f"{example['instruction']}\n\n"
            f"{example['input']}"
        )
    else:
        user_content = example['instruction']

    messages = [
        {
            "role":"user",
            "content": user_content,
        },
        {
            "role":"assistant",
            "content": example["output"],
        },
    ]

    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text":chat
    }

    