"""
Train GPT-2 Medium models with emotion conditioning.

This script trains Token and Prefix conditioning methods using GPT-2 Medium (355M params)
with gradient checkpointing to fit in 8GB VRAM.

Expected improvements over GPT-2 Small:
- Token: 30% -> 35-40%
- Prefix: 38% -> 43-48%

Training time: ~10 hours total on RTX 4060 Mobile
"""

import os
import torch
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from utils.dailydialog_processor import load_and_prepare_dailydialog

def train_model(method, base_model='gpt2-medium', epochs=3):
    """Train a single model with gradient checkpointing for memory efficiency."""
    
    print(f"\n{'='*80}")
    print(f"Training {method.upper()} model with GPT-2 Medium")
    print(f"{'='*80}\n")
    
    # Load model and tokenizer
    print("Loading GPT-2 Medium model...")
    tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    model = GPT2LMHeadModel.from_pretrained(base_model)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    print("✓ Gradient checkpointing enabled (saves ~40% VRAM)")
    
    # Add special tokens for emotion conditioning
    special_tokens = {
        'additional_special_tokens': [
            '<NEUTRAL>', '<HAPPY>', '<SAD>', '<ANGRY>', 
            '<FEAR>', '<DISGUST>', '<SURPRISE>'
        ],
        'pad_token': '<PAD>'
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Load and prepare dataset
    print(f"Loading DailyDialog dataset for {method} method...")
    datasets_raw = load_and_prepare_dailydialog(method=method, use_context=True)
    
    # Convert lists to HuggingFace Dataset objects
    datasets = {
        split: Dataset.from_dict({'text': [item['text'] for item in data]})
        for split, data in datasets_raw.items()
    }
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=128,  # Reduced from 256 to save memory
            padding='max_length'
        )
    
    print("Tokenizing datasets...")
    tokenized_datasets = {
        split: dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split}"
        )
        for split, dataset in datasets.items()
    }
    
    # Training arguments optimized for 8GB VRAM
    output_dir = f'./models/gpt2medium_{method}'
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,      # Reduced from 8 (Medium is 3x larger)
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,       # Effective batch size = 2*4 = 8
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs/gpt2medium_{method}',
        logging_steps=100,
        eval_strategy='steps',
        eval_steps=500,
        save_strategy='steps',
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=torch.cuda.is_available(),      # Mixed precision training
        gradient_checkpointing=True,         # Enable in training args too
        optim='adamw_torch',                 # Memory-efficient optimizer
        dataloader_pin_memory=False,         # Reduce memory pressure
        report_to='none'
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
    )
    
    # Train
    print(f"\nStarting training... (this will take ~5 hours)")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Total steps: ~{len(tokenized_datasets['train']) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * epochs}")
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets['test'])
    print(f"Test perplexity: {torch.exp(torch.tensor(test_results['eval_loss'])):.2f}")
    
    print(f"\n✓ {method.upper()} model training complete!")
    print(f"Model saved to: {output_dir}")
    
    # Clear memory
    del model
    del trainer
    torch.cuda.empty_cache()
    
    return output_dir

def main():
    """Train both Token and Prefix models with GPT-2 Medium."""
    
    print("\n" + "="*80)
    print("GPT-2 MEDIUM TRAINING SCRIPT")
    print("="*80)
    print("\nThis will train 2 models:")
    print("  1. Token conditioning (gpt2medium_token)")
    print("  2. Prefix conditioning (gpt2medium_prefix)")
    print("\nExpected time: ~10 hours total")
    print("VRAM usage: ~7.5GB (fits in 8GB with gradient checkpointing)")
    print("="*80 + "\n")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will be very slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {gpu_name}")
        print(f"GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 7.5:
            print("\nWARNING: Your GPU has less than 7.5GB VRAM.")
            print("Training may fail with OOM errors.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Training cancelled.")
                return
    
    # Train both models
    methods = ['token', 'prefix']
    
    for i, method in enumerate(methods, 1):
        print(f"\n{'#'*80}")
        print(f"# Training {i}/2: {method.upper()} method")
        print(f"{'#'*80}\n")
        
        try:
            output_dir = train_model(method)
            print(f"\n✓ Successfully trained {method} model")
            print(f"  Location: {output_dir}")
        except Exception as e:
            print(f"\n✗ Error training {method} model:")
            print(f"  {str(e)}")
            print("\nContinuing to next model...")
            continue
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nTrained models:")
    print("  - ./models/gpt2medium_token")
    print("  - ./models/gpt2medium_prefix")
    print("\nNext steps:")
    print("  1. Run evaluation: jupyter notebook notebooks/03_evaluation.ipynb")
    print("  2. Compare with GPT-2 Small results")
    print("  3. Try ensemble prediction for even better results")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
