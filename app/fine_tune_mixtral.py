# scripts/fine_tune_mixtral.py

import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DataCollatorWithPadding
from dataset import Dataset
import torch

def load_dataset(path):
    df = pd.read_csv(path)
    dataset = Dataset.from_pandas(df)
    return dataset

def preprocess_function(examples, tokenizer):
    return tokenizer(examples['question'], examples['answer'], truncation=True, padding='max_length')

def main():
    # Configuration
    model_name = "Mixtal/8x22b"  # Replace with the correct model name
    dataset_path = "data/qa_dataset.csv"
    output_dir = "models/fine_tuned_mixtral"
    epochs = 3
    batch_size = 8

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Load and preprocess dataset
    raw_dataset = load_dataset(dataset_path)
    tokenized_dataset = raw_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    # Start fine-tuning
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    main()
