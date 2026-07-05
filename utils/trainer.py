from trl import SFTTrainer
from configs.train_config import lora_config,training_args

def get_trainer(
    model,
    tokenizer,
    dataset,
    args,
    peft_config,
):
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    return trainer