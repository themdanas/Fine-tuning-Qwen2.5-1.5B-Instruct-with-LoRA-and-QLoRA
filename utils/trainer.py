from trl import SFTTrainer

def get_trainer(
    model,
    tokenizer,
    dataset,
    training_args,
    lora_config,
):
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    return trainer