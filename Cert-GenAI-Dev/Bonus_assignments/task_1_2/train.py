
import argparse
import os
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--training-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load dataset
    data_path = os.path.join(args.training_dir, "financial_qa_dataset.csv")
    df = pd.read_csv(data_path)

    # Prepare dataset
    def format_instruction(row):
        return f"Question: {row['question']}\nAnswer: {row['answer']}"

    df["text"] = df.apply(format_instruction, axis=1)
    dataset = Dataset.from_pandas(df[["text"]])

    # Load model and tokenizer
    model_name = "distilgpt2"  # Use a smaller model for example purposes
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
