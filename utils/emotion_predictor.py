"""
Emotion Predictor

Utility for loading trained RoBERTa emotion classifier and predicting emotions from text.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Import emotion mappings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.emotion_mapping import TARGET_EMOTIONS


class EmotionPredictor:
    """
    Wrapper class for emotion prediction using fine-tuned RoBERTa model.
    """
    
    def __init__(self, model_path, device=None):
        """
        Initialize emotion predictor.
        
        Args:
            model_path (str): Path to fine-tuned RoBERTa model
            device (str, optional): Device to run model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading emotion classifier from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.emotions = TARGET_EMOTIONS
        print(f"Loaded model with {len(self.emotions)} emotion categories")
    
    def predict_emotion(self, text, return_all_scores=False):
        """
        Predict emotion from input text.
        
        Args:
            text (str): Input text to classify
            return_all_scores (bool): If True, return scores for all emotions
            
        Returns:
            str or dict: Predicted emotion label, or dict of all scores if return_all_scores=True
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Get predicted class
        predicted_idx = torch.argmax(probs, dim=-1).item()
        predicted_emotion = self.emotions[predicted_idx]
        
        if return_all_scores:
            scores = {
                emotion: probs[0][i].item()
                for i, emotion in enumerate(self.emotions)
            }
            return {
                'emotion': predicted_emotion,
                'confidence': probs[0][predicted_idx].item(),
                'all_scores': scores
            }
        
        return predicted_emotion
    
    def predict_batch(self, texts, batch_size=32):
        """
        Predict emotions for a batch of texts.
        
        Args:
            texts (list): List of input texts
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of predicted emotion labels
        """
        predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_indices = torch.argmax(logits, dim=-1)
            
            # Convert to emotion labels
            batch_predictions = [
                self.emotions[idx] for idx in predicted_indices.cpu().tolist()
            ]
            predictions.extend(batch_predictions)
        
        return predictions


def predict_emotion(text, model_path=None):
    """
    Convenience function to predict emotion from text.
    
    Args:
        text (str): Input text
        model_path (str, optional): Path to model (defaults to standard location)
        
    Returns:
        str: Predicted emotion label
    """
    if model_path is None:
        # Default path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'emotion_classifier_roberta')
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model not found at {model_path}. Please train the model first.")
    
    predictor = EmotionPredictor(model_path)
    return predictor.predict_emotion(text)


if __name__ == "__main__":
    # Test predictions (requires trained model)
    test_texts = [
        "I am so happy and excited about this!",
        "This makes me really angry and frustrated.",
        "I feel sad and disappointed.",
        "That's scary and frightening.",
        "This is disgusting and revolting.",
        "Wow, what a surprise!",
        "The weather is okay today."
    ]
    
    try:
        predictor = EmotionPredictor("../models/emotion_classifier_roberta")
        print("=== Emotion Predictions ===")
        for text in test_texts:
            result = predictor.predict_emotion(text, return_all_scores=True)
            print(f"\nText: {text}")
            print(f"Predicted: {result['emotion']} (confidence: {result['confidence']:.3f})")
    except Exception as e:
        print(f"Error: {e}")
        print("Please train the emotion classifier first using notebook 01.")
