"""
Emotion Mapping Utilities

Maps GoEmotions 27-category labels to 7 target emotions and handles DailyDialog emotion IDs.

References:
    - GoEmotions: https://github.com/google-research/google-research/tree/master/goemotions
    - DailyDialog: https://aclanthology.org/I17-1099/
"""

# Target 7 emotions for this project
TARGET_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

# GoEmotions original 27 emotions + neutral
GOEMOTIONS_EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Mapping from GoEmotions 27 categories to our 7 target emotions
GOEMOTIONS_TO_TARGET = {
    # Happy category
    'admiration': 'happy',
    'amusement': 'happy',
    'approval': 'happy',
    'excitement': 'happy',
    'gratitude': 'happy',
    'joy': 'happy',
    'love': 'happy',
    'optimism': 'happy',
    'pride': 'happy',
    'relief': 'happy',
    'caring': 'happy',
    
    # Sad category
    'sadness': 'sad',
    'disappointment': 'sad',
    'grief': 'sad',
    'remorse': 'sad',
    
    # Angry category
    'anger': 'angry',
    'annoyance': 'angry',
    'disapproval': 'angry',
    
    # Fear category
    'fear': 'fear',
    'nervousness': 'fear',
    
    # Disgust category
    'disgust': 'disgust',
    'embarrassment': 'disgust',
    
    # Surprise category
    'surprise': 'surprise',
    'confusion': 'surprise',
    'curiosity': 'surprise',
    'realization': 'surprise',
    
    # Neutral category
    'neutral': 'neutral',
    'desire': 'neutral',  # Ambiguous, default to neutral
}

# DailyDialog emotion ID to label mapping (0-6)
DAILYDIALOG_ID_TO_LABEL = {
    0: 'neutral',
    1: 'angry',
    2: 'disgust',
    3: 'fear',
    4: 'happy',
    5: 'sad',
    6: 'surprise'
}

# Reverse mapping
DAILYDIALOG_LABEL_TO_ID = {v: k for k, v in DAILYDIALOG_ID_TO_LABEL.items()}


def map_goemotions_to_target(emotion_labels):
    """
    Map GoEmotions 27-category labels to 7 target emotions.
    
    Args:
        emotion_labels (list): List of emotion labels from GoEmotions (can be multiple per sample)
        
    Returns:
        str: Single target emotion label (if multiple, returns first mapped emotion)
    """
    if not emotion_labels or len(emotion_labels) == 0:
        return 'neutral'
    
    # Map all labels and return the first valid one
    for label in emotion_labels:
        if label in GOEMOTIONS_TO_TARGET:
            return GOEMOTIONS_TO_TARGET[label]
    
    # Default to neutral if no mapping found
    return 'neutral'


def get_target_emotion_id(emotion_label):
    """
    Get numeric ID for target emotion label (0-6).
    
    Args:
        emotion_label (str): Target emotion label
        
    Returns:
        int: Numeric emotion ID
    """
    if emotion_label in TARGET_EMOTIONS:
        return TARGET_EMOTIONS.index(emotion_label)
    return 0  # Default to neutral


def dailydialog_id_to_label(emotion_id):
    """
    Convert DailyDialog numeric emotion ID to label.
    
    Args:
        emotion_id (int): Emotion ID (0-6)
        
    Returns:
        str: Emotion label
    """
    return DAILYDIALOG_ID_TO_LABEL.get(emotion_id, 'neutral')


def get_emotion_token(emotion_label):
    """
    Get special token for emotion conditioning (Method C).
    
    Args:
        emotion_label (str): Emotion label
        
    Returns:
        str: Special token (e.g., '<HAPPY>')
    """
    return f"<{emotion_label.upper()}>"


def get_emotion_prefix(emotion_label):
    """
    Get text prefix for emotion conditioning (Method B).
    
    Args:
        emotion_label (str): Emotion label
        
    Returns:
        str: Prefix string (e.g., 'happy: ')
    """
    return f"{emotion_label}: "


# Create list of all special emotion tokens for tokenizer extension
EMOTION_TOKENS = [get_emotion_token(emotion) for emotion in TARGET_EMOTIONS]


if __name__ == "__main__":
    # Test mappings
    print("Target Emotions:", TARGET_EMOTIONS)
    print("\nEmotion Tokens:", EMOTION_TOKENS)
    
    # Test GoEmotions mapping
    print("\n=== GoEmotions Mapping Test ===")
    test_labels = [['joy'], ['anger', 'annoyance'], ['neutral'], ['curiosity']]
    for labels in test_labels:
        mapped = map_goemotions_to_target(labels)
        print(f"{labels} -> {mapped}")
    
    # Test DailyDialog mapping
    print("\n=== DailyDialog Mapping Test ===")
    for i in range(7):
        label = dailydialog_id_to_label(i)
        print(f"ID {i} -> {label}")
        print(f"  Token: {get_emotion_token(label)}")
        print(f"  Prefix: {get_emotion_prefix(label)}")
