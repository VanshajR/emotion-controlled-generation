"""
DailyDialog Dataset Processor

Processes DailyDialog dataset and prepares it for different GPT-2 conditioning methods.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from utils.emotion_mapping import (
    dailydialog_id_to_label,
    get_emotion_prefix,
    get_emotion_token,
    TARGET_EMOTIONS
)


def extract_utterance_emotion_pairs(dataset, split='train'):
    """
    Extract (utterance, emotion) pairs from DailyDialog dataset.
    
    Args:
        dataset: Loaded DailyDialog dataset
        split (str): Dataset split ('train', 'validation', 'test')
        
    Returns:
        list: List of dicts with 'utterance' and 'emotion' keys
    """
    pairs = []
    
    for sample in dataset[split]:
        dialog = sample['utterances']  # Changed from 'dialog' to 'utterances'
        emotions = sample['emotions']  # Changed from 'emotion' to 'emotions'
        
        # Extract each utterance with its emotion
        for utterance, emotion_id in zip(dialog, emotions):
            emotion_label = dailydialog_id_to_label(emotion_id)
            pairs.append({
                'utterance': utterance,
                'emotion': emotion_label
            })
    
    return pairs


def create_context_response_pairs(dataset, split='train', use_context=True):
    """
    Create (context, response, emotion) triplets from dialogs.
    
    Args:
        dataset: Loaded DailyDialog dataset
        split (str): Dataset split
        use_context (bool): If True, use previous utterance as context
        
    Returns:
        list: List of dicts with 'context', 'response', and 'emotion' keys
    """
    triplets = []
    
    for sample in dataset[split]:
        dialog = sample['utterances']  # Changed from 'dialog' to 'utterances'
        emotions = sample['emotions']  # Changed from 'emotion' to 'emotions'
        
        # Create pairs: previous utterance -> current utterance
        for i in range(1, len(dialog)):
            context = dialog[i - 1] if use_context else ""
            response = dialog[i]
            emotion_id = emotions[i]
            emotion_label = dailydialog_id_to_label(emotion_id)
            
            triplets.append({
                'context': context,
                'response': response,
                'emotion': emotion_label
            })
    
    return triplets


def prepare_baseline_dataset(triplets):
    """
    Method A: Baseline - plain text without emotion conditioning.
    
    Args:
        triplets (list): List of context-response-emotion triplets
        
    Returns:
        list: List of formatted training texts
    """
    formatted = []
    
    for item in triplets:
        if item['context']:
            text = f"{item['context']}\n{item['response']}"
        else:
            text = item['response']
        
        formatted.append({
            'text': text,
            'emotion': item['emotion']
        })
    
    return formatted


def prepare_prefix_dataset(triplets):
    """
    Method B: Prefix-based conditioning (e.g., "happy: response text").
    
    Args:
        triplets (list): List of context-response-emotion triplets
        
    Returns:
        list: List of formatted training texts with emotion prefixes
    """
    formatted = []
    
    for item in triplets:
        emotion_prefix = get_emotion_prefix(item['emotion'])
        
        if item['context']:
            text = f"{item['context']}\n{emotion_prefix}{item['response']}"
        else:
            text = f"{emotion_prefix}{item['response']}"
        
        formatted.append({
            'text': text,
            'emotion': item['emotion']
        })
    
    return formatted


def prepare_token_dataset(triplets):
    """
    Method C: Special token conditioning - single token at start for clarity.
    Format: <EMOTION> context\nresponse
    
    Args:
        triplets (list): List of context-response-emotion triplets
        
    Returns:
        list: List of formatted training texts with emotion tokens
    """
    formatted = []
    
    for item in triplets:
        emotion_token = get_emotion_token(item['emotion'])
        
        # Single emotion token at start for clear conditioning
        if item['context']:
            text = f"{emotion_token} {item['context']}\n{item['response']}"
        else:
            text = f"{emotion_token} {item['response']}"
        
        formatted.append({
            'text': text,
            'emotion': item['emotion']
        })
    
    return formatted


def prepare_lora_dataset(triplets):
    """
    Method D: LoRA with special tokens (same format as Method C but for LoRA training).
    
    Args:
        triplets (list): List of context-response-emotion triplets
        
    Returns:
        list: List of formatted training texts
    """
    # Same format as token-based, but will be trained with LoRA
    return prepare_token_dataset(triplets)


def load_and_prepare_dailydialog(method='baseline', use_context=True, cache_dir=None):
    """
    Load DailyDialog and prepare for specified training method.
    
    Args:
        method (str): Training method ('baseline', 'prefix', 'token', 'lora')
        use_context (bool): Whether to include context utterances
        cache_dir (str): Directory to cache dataset
        
    Returns:
        dict: Dictionary with 'train', 'validation', 'test' splits
    """
    print(f"Loading DailyDialog dataset...")
    dataset = load_dataset("roskoN/dailydialog", cache_dir=cache_dir)
    
    print(f"Extracting context-response pairs...")
    train_triplets = create_context_response_pairs(dataset, 'train', use_context)
    val_triplets = create_context_response_pairs(dataset, 'validation', use_context)
    test_triplets = create_context_response_pairs(dataset, 'test', use_context)
    
    print(f"Train samples: {len(train_triplets)}")
    print(f"Validation samples: {len(val_triplets)}")
    print(f"Test samples: {len(test_triplets)}")
    
    # Prepare according to method
    if method == 'baseline':
        prepare_fn = prepare_baseline_dataset
    elif method == 'prefix':
        prepare_fn = prepare_prefix_dataset
    elif method == 'token':
        prepare_fn = prepare_token_dataset
    elif method == 'lora':
        prepare_fn = prepare_lora_dataset
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Preparing data for method: {method}")
    prepared_data = {
        'train': prepare_fn(train_triplets),
        'validation': prepare_fn(val_triplets),
        'test': prepare_fn(test_triplets)
    }
    
    # Print sample
    print(f"\n=== Sample from {method} dataset ===")
    print(prepared_data['train'][0]['text'])
    print(f"Emotion: {prepared_data['train'][0]['emotion']}")
    
    return prepared_data


def get_emotion_distribution(data):
    """
    Get distribution of emotions in dataset.
    
    Args:
        data (list): List of data samples with 'emotion' field
        
    Returns:
        dict: Emotion counts
    """
    emotion_counts = {emotion: 0 for emotion in TARGET_EMOTIONS}
    
    for item in data:
        emotion = item['emotion']
        emotion_counts[emotion] += 1
    
    return emotion_counts


if __name__ == "__main__":
    # Test data processing
    print("=== Testing DailyDialog Processing ===\n")
    
    for method in ['baseline', 'prefix', 'token']:
        print(f"\n{'='*60}")
        print(f"Method: {method.upper()}")
        print('='*60)
        
        data = load_and_prepare_dailydialog(method=method, use_context=True)
        
        # Show emotion distribution
        print("\nEmotion distribution (train):")
        dist = get_emotion_distribution(data['train'])
        for emotion, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {count} ({count/len(data['train'])*100:.1f}%)")
        
        # Show more samples
        print(f"\nFirst 3 samples:")
        for i, sample in enumerate(data['train'][:3]):
            print(f"\n{i+1}. [{sample['emotion']}]")
            print(f"   {sample['text'][:100]}...")
