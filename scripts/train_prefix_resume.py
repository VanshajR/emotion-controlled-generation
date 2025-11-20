"""
Resume Prefix model training with reduced batch size
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import sys
sys.path.append('.')
from utils.dailydialog_processor import load_and_prepare_dailydialog
from utils.emotion_mapping import EMOTION_TOKENS

print("="*80)
print("RESUME PREFIX MODEL TRAINING")
print("="*80)
print("\nReducing batch size from 2 to 1 to prevent OOM")
print("Gradient accumulation increased from 4 to 8 (same effective batch)")
print("="*80 + "\n")

# Model settings
MODEL_NAME = 'gpt2-medium'
OUTPUT_DIR = './models/gpt2medium_prefix'
METHOD = 'prefix'

# Check if checkpoint exists
if not os.path.exists(OUTPUT_DIR):
    print(f"❌ No checkpoint found at {OUTPUT_DIR}")
    print("Run train_medium_models.py first to create initial checkpoint")
    exit(1)

# Find latest checkpoint
checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith('checkpoint-')]
if not checkpoints:
    print("❌ No checkpoints found. Model may have saved final state already.")
    print("Check if training already completed.")
    exit(1)

latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
checkpoint_path = os.path.join(OUTPUT_DIR, latest_checkpoint)
print(f"📂 Resuming from: {checkpoint_path}\n")

# Load model from checkpoint FIRST (keeps original vocab size)
print("Loading model from checkpoint...")
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
checkpoint_vocab_size = model.transformer.wte.weight.shape[0]
print(f"✓ Model embedding size: {checkpoint_vocab_size}\n")

# Load tokenizer and match it to checkpoint's vocab size
print("Loading tokenizer from base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Add emotion tokens to match checkpoint
print("Adding emotion tokens to tokenizer...")
special_tokens = {'additional_special_tokens': EMOTION_TOKENS}
num_added = tokenizer.add_special_tokens(special_tokens)
print(f"✓ Added {num_added} emotion tokens")
print(f"✓ Tokenizer vocab size: {len(tokenizer)}")

# Verify sizes match (checkpoint should be 1 larger due to training script quirk)
if checkpoint_vocab_size != len(tokenizer):
    print(f"\n⚠️  Size mismatch: checkpoint has {checkpoint_vocab_size}, tokenizer has {len(tokenizer)}")
    print(f"   This is normal - checkpoint may have one extra embedding from training.")
    print(f"   Keeping checkpoint's original size of {checkpoint_vocab_size}\n")

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
print("✓ Gradient checkpointing enabled\n")

# Prepare dataset
print("Loading DailyDialog dataset...")
data_dict = load_and_prepare_dailydialog(method=METHOD, use_context=True)
train_data = data_dict['train']
val_data = data_dict['validation']

print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}\n")

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_dict({
    'text': [item['text'] for item in train_data],
    'emotion': [item['emotion'] for item in train_data]
})
val_dataset = Dataset.from_dict({
    'text': [item['text'] for item in val_data],
    'emotion': [item['emotion'] for item in val_data]
})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=128,
        padding='max_length'
    )

print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['emotion'])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['emotion'])
print("✓ Tokenization complete\n")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# REDUCED training arguments - batch_size=1 instead of 2
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=False,  # Don't overwrite - we're resuming!
    num_train_epochs=3,
    per_device_train_batch_size=1,  # REDUCED from 2
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # INCREASED from 4 (same effective batch)
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    eval_strategy='steps',
    eval_steps=500,
    fp16=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    report_to='none'
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Resume training
print("="*80)
print("RESUMING TRAINING")
print("="*80)
print(f"Batch size: 1 (down from 2)")
print(f"Gradient accumulation: 8 steps (up from 4)")
print(f"Effective batch size: 8 (same as before)")
print(f"Resuming from step: {latest_checkpoint}")
print("="*80 + "\n")

try:
    trainer.train(resume_from_checkpoint=checkpoint_path)
    print("\n✅ Training completed successfully!")
    
    # Save final model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    print("\nIf still OOM, try:")
    print("1. Reduce max_length to 96")
    print("2. Increase gradient_accumulation_steps to 16")
    print("3. Close all other GPU applications")
