import os
import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


def make_format_prompt(tokenizer):
    def format_prompt(samples):
        # SFTTrainer passes a batch: each value is a list
        results = []
        for instruction, output in zip(samples["instruction"], samples["output"]):
            messages = [{"role": "user", "content": instruction}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            results.append(prompt + output + tokenizer.eos_token)
        return results
    return format_prompt


def train():
    with open("training/config.yaml") as f:
        config = yaml.safe_load(f)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {config['model']['name']}...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=config["qlora"]["r"],
        lora_alpha=config["qlora"]["lora_alpha"],
        lora_dropout=config["qlora"]["lora_dropout"],
        target_modules=config["qlora"]["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    format_prompt = make_format_prompt(tokenizer)

    train_data = load_dataset(
        "json", data_files="data/processed/train.jsonl", split="train"
    )
    val_data = load_dataset(
        "json", data_files="data/processed/val.jsonl", split="train"
    )

    training_args = SFTConfig(
        output_dir=config["output"]["dir"],
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation"],
        learning_rate=config["training"]["learning_rate"],
        fp16=True,
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
        logging_steps=config["training"]["logging_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        max_seq_length=config["model"]["max_length"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        formatting_func=format_prompt,
    )

    print("Starting training...")
    trainer.train()

    output_dir = config["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete.")
    print(f"LoRA adapters saved to {output_dir}")


if __name__ == "__main__":
    train()
