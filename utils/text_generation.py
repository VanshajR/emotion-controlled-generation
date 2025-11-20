"""
Text Generation Utilities

Functions for generating text from fine-tuned GPT-2 models with emotion conditioning.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.emotion_mapping import get_emotion_token, get_emotion_prefix, TARGET_EMOTIONS


class EmotionControlledGenerator:
    """
    Wrapper for emotion-controlled text generation with GPT-2 variants.
    """
    
    def __init__(self, model_path, conditioning_method='baseline', device=None):
        """
        Initialize generator.
        
        Args:
            model_path (str): Path to fine-tuned GPT-2 model
            conditioning_method (str): One of 'baseline', 'prefix', 'token', 'lora'
            device (str, optional): Device to run on
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.conditioning_method = conditioning_method
        
        print(f"Loading model from: {model_path}")
        print(f"Conditioning method: {conditioning_method}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        print("Model loaded successfully")
    
    def generate_response(
        self,
        context,
        target_emotion,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True
    ):
        """
        Generate emotion-controlled response.
        
        Args:
            context (str): Input context/prompt
            target_emotion (str): Target emotion for generation
            max_length (int): Maximum generation length
            num_return_sequences (int): Number of responses to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling parameter
            top_p (float): Top-p (nucleus) sampling parameter
            do_sample (bool): Whether to use sampling
            
        Returns:
            list: Generated responses
        """
        # Prepare input based on conditioning method
        if self.conditioning_method == 'baseline':
            prompt = f"{context}\n" if context else ""
        elif self.conditioning_method == 'prefix':
            prefix = get_emotion_prefix(target_emotion)
            prompt = f"{context}\n{prefix}" if context else prefix
        elif self.conditioning_method in ['token', 'lora']:
            token = get_emotion_token(target_emotion)
            # Match training format: <EMOTION> context\n
            prompt = f"{token} {context}\n" if context else f"{token} "
        else:
            raise ValueError(f"Unknown conditioning method: {self.conditioning_method}")
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        responses = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            
            # Remove prompt from response
            if self.conditioning_method == 'prefix':
                prefix = get_emotion_prefix(target_emotion)
                if prefix in text:
                    # Extract text after prefix
                    parts = text.split(prefix, 1)
                    if len(parts) > 1:
                        text = parts[1].strip()
            elif self.conditioning_method in ['token', 'lora']:
                # Remove emotion token if present
                for emotion in TARGET_EMOTIONS:
                    token = get_emotion_token(emotion)
                    text = text.replace(token, "").strip()
            
            # Remove context if present
            if context and text.startswith(context):
                text = text[len(context):].strip()
            
            responses.append(text)
        
        return responses
    
    def generate_batch(
        self,
        contexts,
        target_emotions,
        max_length=100,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    ):
        """
        Generate responses for a batch of contexts and emotions.
        
        Args:
            contexts (list): List of context strings
            target_emotions (list): List of target emotions
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling
            top_p (float): Top-p sampling
            
        Returns:
            list: Generated responses
        """
        responses = []
        
        for context, emotion in zip(contexts, target_emotions):
            response = self.generate_response(
                context=context,
                target_emotion=emotion,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True
            )[0]
            responses.append(response)
        
        return responses


def generate_with_all_emotions(
    model_path,
    context,
    conditioning_method='baseline',
    max_length=100,
    temperature=0.7
):
    """
    Generate responses for all 7 emotions.
    
    Args:
        model_path (str): Path to model
        context (str): Input context
        conditioning_method (str): Conditioning method
        max_length (int): Max generation length
        temperature (float): Sampling temperature
        
    Returns:
        dict: Dictionary mapping emotions to generated responses
    """
    generator = EmotionControlledGenerator(model_path, conditioning_method)
    results = {}
    
    print(f"\nGenerating responses for all emotions...")
    print(f"Context: {context}\n")
    
    for emotion in TARGET_EMOTIONS:
        response = generator.generate_response(
            context=context,
            target_emotion=emotion,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1
        )[0]
        results[emotion] = response
        print(f"[{emotion.upper()}] {response}")
    
    return results


if __name__ == "__main__":
    # Test generation (requires trained model)
    test_contexts = [
        "How was your day?",
        "I have some news to share.",
        "What do you think about this?"
    ]
    
    model_variants = {
        'baseline': '../models/gpt2_baseline',
        'prefix': '../models/gpt2_prefix',
        'token': '../models/gpt2_tokens',
        'lora': '../models/gpt2_tokens_lora'
    }
    
    print("=== Emotion-Controlled Generation Test ===\n")
    
    for method, model_path in model_variants.items():
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Method: {method.upper()}")
        print('='*70)
        
        try:
            results = generate_with_all_emotions(
                model_path=model_path,
                context=test_contexts[0],
                conditioning_method=method,
                max_length=50,
                temperature=0.7
            )
        except Exception as e:
            print(f"Error: {e}")
