from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("Uploading GPT-2 Prefix-Small (Best Model) to HuggingFace")
print("=" * 60)

print("\n[1/4] Loading GPT-2 Prefix-Small model...")
print("This is our best model with 38.2% emotion accuracy!")
model = AutoModelForCausalLM.from_pretrained(
    "d:/Auralis/models/gpt2_prefix_v2"
)
tokenizer = AutoTokenizer.from_pretrained(
    "d:/Auralis/models/gpt2_prefix_v2"
)
print("✅ Model and tokenizer loaded successfully")

print("\n[2/4] Pushing model to HuggingFace Hub...")
print("This may take 3-5 minutes (model is ~500MB)...")
model.push_to_hub("VanshajR/gpt2-emotion-prefix")
print("✅ Model uploaded")

print("\n[3/4] Pushing tokenizer to HuggingFace Hub...")
tokenizer.push_to_hub("VanshajR/gpt2-emotion-prefix")
print("✅ Tokenizer uploaded")

print("\n[4/4] Upload complete!")
print("=" * 60)
print("✅ GPT-2 Prefix-Small uploaded successfully!")
print(f"🔗 Model URL: https://huggingface.co/VanshajR/gpt2-emotion-prefix")
print("\n📊 Model Performance:")
print("   • Emotion Accuracy: 38.2%")
print("   • Improvement: +9.8pp over baseline")
print("   • Test Set: 6,740 samples")
print("\nNext steps:")
print("1. Go to the model page and edit the model card")
print("2. Add model description, performance metrics, and usage examples")
print("3. Test the model using the inference widget")
print("=" * 60)
