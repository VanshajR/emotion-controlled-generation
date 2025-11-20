from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("=" * 60)
print("Uploading RoBERTa Emotion Classifier to HuggingFace")
print("=" * 60)

print("\n[1/4] Loading emotion classifier...")
model = AutoModelForSequenceClassification.from_pretrained(
    "d:/Auralis/models/emotion_classifier_roberta"
)
tokenizer = AutoTokenizer.from_pretrained(
    "d:/Auralis/models/emotion_classifier_roberta"
)
print("✅ Model and tokenizer loaded successfully")

print("\n[2/4] Pushing model to HuggingFace Hub...")
print("This may take 2-3 minutes depending on your internet speed...")
model.push_to_hub("your-username/roberta-emotion-7class")
print("✅ Model uploaded")

print("\n[3/4] Pushing tokenizer to HuggingFace Hub...")
tokenizer.push_to_hub("your-username/roberta-emotion-7class")
print("✅ Tokenizer uploaded")

print("\n[4/4] Upload complete!")
print("=" * 60)
print("✅ Emotion Classifier uploaded successfully!")
print(f"🔗 Model URL: https://huggingface.co/your-username/roberta-emotion-7class")
print("\nNext steps:")
print("1. Go to the model page and edit the model card")
print("2. Add model description, metrics, and usage examples")
print("3. Verify the model works by testing the inference widget")
print("=" * 60)
