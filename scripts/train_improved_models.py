"""
Quick training script for improved Token and Prefix models.
Run this instead of notebook 02 for faster execution.
"""

import os
import sys
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

sys.path.append('.')
from utils.emotion_mapping import EMOTION_TOKENS
from utils.dailydialog_processor import load_and_prepare_dailydialog

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

def prepare_tokenizer(method, base_model_name='gpt2'):
    """Use GPT-2 instead of DialoGPT for better baseline performance"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    if method in ['token']:
        num_added = tokenizer.add_special_tokens({'additional_special_tokens': EMOTION_TOKENS})
        print(f"  Added {num_added} emotion tokens")
    
    return tokenizer


def prepare_model(method, tokenizer, base_model_name='gpt2'):
    """Use GPT-2 instead of DialoGPT for better baseline performance"""
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    if method in ['token']:
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Resized embeddings to {len(tokenizer)} tokens")
    
    return model


def tokenize_dataset(data, tokenizer, max_length=256):
    texts = [sample['text'] for sample in data]
    
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False
    )
    
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask']
    })
    
    return dataset


def train_model(method, train_data, val_data):
    print(f"\n{'='*70}")
    print(f"Training {method.upper()} with DialoGPT-small")
    print('='*70)
    
    # Prepare
    print("\n1. Preparing tokenizer...")
    tokenizer = prepare_tokenizer(method)
    
    print("\n2. Preparing model...")
    model = prepare_model(method, tokenizer)
    
    print("\n3. Tokenizing datasets...")
    train_dataset = tokenize_dataset(train_data, tokenizer)
    val_dataset = tokenize_dataset(val_data, tokenizer)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments - optimized for emotion control
    output_dir = f"models/gpt2_{method}_v2"  # v2 to distinguish from broken models
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Reduced from 5 to avoid overfitting
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,  # Slightly higher for faster convergence
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",
        seed=42,
        fp16=torch.cuda.is_available()
    )
    
    print("\n4. Training configuration:")
    print(f"  Epochs: 5")
    print(f"  Batch size: 8")
    print(f"  Learning rate: 2e-5")
    print(f"  Warmup steps: 1000")
    print(f"  Output: {output_dir}")
    print(f"  FP16: {training_args.fp16}")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    print("\n5. Starting training...")
    print("="*70)
    
    # Train
    train_result = trainer.train()
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    
    # Display training metrics
    print("\n=== Training Metrics ===")
    for key, value in train_result.metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save
    print(f"\n6. Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n✅ Model saved successfully!\n")
    
    return trainer


if __name__ == "__main__":
    # Load datasets
    print("Loading DailyDialog datasets...\n")
    
    # Token dataset
    print("Loading Token dataset...")
    token_data = load_and_prepare_dailydialog(method='token', use_context=True)
    
    # Prefix dataset  
    print("Loading Prefix dataset...")
    prefix_data = load_and_prepare_dailydialog(method='prefix', use_context=True)
    
    print("\n" + "="*70)
    print("TRAINING TOKEN MODEL (Estimated: 3-3.5 hours)")
    print("="*70)
    train_model('token', token_data['train'], token_data['validation'])
    
    print("\n" + "="*70)
    print("TRAINING PREFIX MODEL (Estimated: 3-3.5 hours)")
    print("="*70)
    train_model('prefix', prefix_data['train'], prefix_data['validation'])
    
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run notebook 03_evaluation.ipynb")
    print("2. Update model paths to use '_improved' versions")
    print("3. Check new metrics!")
