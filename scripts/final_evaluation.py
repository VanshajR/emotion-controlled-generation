"""
FINAL COMPREHENSIVE EVALUATION

Evaluates all models on the FULL test set for publication-ready results.
Generates all necessary plots and statistics for the report.

Runtime: ~45 minutes on full test set (6740 samples)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from utils.emotion_predictor import EmotionPredictor
from utils.text_generation import EmotionControlledGenerator
from utils.dailydialog_processor import load_and_prepare_dailydialog
from utils.emotion_mapping import TARGET_EMOTIONS

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

def extract_context_and_reference(text_sample):
    """Extract context and reference from sample."""
    if '\n' in text_sample:
        parts = text_sample.split('\n', 1)
        return parts[0], parts[1] if len(parts) > 1 else ""
    return "", text_sample

def evaluate_model(model_path, method, test_data, emotion_classifier, model_name):
    """Evaluate a single model on test data."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None
    
    # Generate responses
    print("Generating responses...")
    generations = []
    contexts = []
    references = []
    target_emotions = []
    
    generator = EmotionControlledGenerator(model_path, conditioning_method=method)
    
    for sample in tqdm(test_data, desc=f"{model_name}"):
        context, reference = extract_context_and_reference(sample['text'])
        emotion = sample['emotion']
        
        contexts.append(context)
        references.append(reference)
        target_emotions.append(emotion)
        
        try:
            response = generator.generate_response(
                context=context,
                target_emotion=emotion,
                max_length=50,
                temperature=0.8,
                top_p=0.9,
                num_return_sequences=1
            )[0]
            
            # Truncate to first sentence if too long
            if len(response) > 150:
                for sep in ['. ', '! ', '? ']:
                    if sep in response:
                        response = response.split(sep)[0] + sep.strip()
                        break
            
            generations.append(response)
        except Exception as e:
            print(f"Error generating: {e}")
            generations.append("")
    
    del generator
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Evaluate emotion accuracy
    print("\nEvaluating emotion accuracy...")
    valid_generations = [g for g in generations if g.strip()]
    valid_targets = [target_emotions[i] for i, g in enumerate(generations) if g.strip()]
    
    predicted_emotions = emotion_classifier.predict_batch(valid_generations)
    
    # Overall accuracy
    correct = sum(1 for pred, target in zip(predicted_emotions, valid_targets) if pred == target)
    accuracy = (correct / len(valid_generations)) * 100 if valid_generations else 0
    
    print(f"✅ Emotion Accuracy: {accuracy:.2f}% ({correct}/{len(valid_generations)})")
    
    # Per-emotion breakdown
    emotion_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for gen, target, pred in zip(generations, target_emotions, 
                                   predicted_emotions if len(predicted_emotions) == len(generations) 
                                   else predicted_emotions + [''] * (len(generations) - len(predicted_emotions))):
        if gen.strip():
            emotion_stats[target]['total'] += 1
            if pred == target:
                emotion_stats[target]['correct'] += 1
    
    # Build results
    results = {
        'model': model_name,
        'overall_accuracy': accuracy,
        'correct': correct,
        'total': len(valid_generations),
        'emotion_breakdown': dict(emotion_stats),
        'generations': generations,
        'targets': target_emotions,
        'predictions': predicted_emotions
    }
    
    return results

def main():
    print("\n" + "="*80)
    print("FINAL COMPREHENSIVE EVALUATION")
    print("="*80)
    print("This will evaluate all models on the FULL test set")
    print("Runtime: ~45 minutes")
    print("="*80 + "\n")
    
    # Load full test set
    print("Loading DailyDialog test set...")
    test_data = load_and_prepare_dailydialog(method='baseline', use_context=True)['test']
    print(f"✓ Loaded {len(test_data)} test samples\n")
    
    # Load emotion classifier
    print("Loading emotion classifier...")
    emotion_classifier = EmotionPredictor('./models/emotion_classifier_roberta')
    print("✓ Classifier loaded\n")
    
    # Models to evaluate
    models_to_evaluate = [
        ('./models/gpt2_baseline', 'baseline', 'Baseline'),
        ('./models/gpt2_prefix_v2', 'prefix', 'Prefix-Small'),
        ('./models/gpt2_token_v2', 'token', 'Token-Small'),
    ]
    
    # Check if Medium models exist
    if os.path.exists('./models/gpt2medium_prefix_final'):
        models_to_evaluate.append(('./models/gpt2medium_prefix_final', 'prefix', 'Prefix-Medium'))
        print("✓ Found Prefix-Medium model - will evaluate")
    else:
        print("⚠️ Prefix-Medium not found - using Small models only")
    
    # Evaluate all models
    all_results = []
    for model_path, method, name in models_to_evaluate:
        result = evaluate_model(model_path, method, test_data, emotion_classifier, name)
        if result:
            all_results.append(result)
    
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Model': result['model'],
            'Emotion_Accuracy': result['overall_accuracy'],
            'Correct': result['correct'],
            'Total': result['total']
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80)
    
    # Calculate improvements
    baseline_acc = df_summary[df_summary['Model'] == 'Baseline']['Emotion_Accuracy'].values[0]
    best_acc = df_summary['Emotion_Accuracy'].max()
    best_model = df_summary[df_summary['Emotion_Accuracy'] == best_acc]['Model'].values[0]
    
    print(f"\n📊 KEY FINDINGS:")
    print(f"  Baseline: {baseline_acc:.2f}%")
    print(f"  Best Model: {best_model} - {best_acc:.2f}%")
    print(f"  Improvement: +{best_acc - baseline_acc:.2f} percentage points")
    print(f"  Relative Improvement: +{((best_acc - baseline_acc) / baseline_acc * 100):.1f}%")
    
    # Save results
    os.makedirs('./results', exist_ok=True)
    df_summary.to_csv('./results/final_evaluation_results.csv', index=False)
    print(f"\n✅ Results saved to ./results/final_evaluation_results.csv")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Overall accuracy comparison
    colors = ['#808080', '#4682b4', '#ff7f50', '#2e8b57']
    bars = ax1.bar(df_summary['Model'], df_summary['Emotion_Accuracy'], 
                    color=colors[:len(df_summary)])
    ax1.set_title('Emotion Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Emotion Accuracy (%)', fontsize=12)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=14.3, color='red', linestyle='--', alpha=0.3, label='Random (14.3%)')
    ax1.legend()
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Per-emotion breakdown for best model
    best_result = [r for r in all_results if r['model'] == best_model][0]
    emotion_names = []
    emotion_accs = []
    
    for emotion in TARGET_EMOTIONS:
        if emotion in best_result['emotion_breakdown']:
            stats = best_result['emotion_breakdown'][emotion]
            if stats['total'] > 0:
                emotion_names.append(emotion.capitalize())
                emotion_accs.append((stats['correct'] / stats['total']) * 100)
    
    ax2.barh(emotion_names, emotion_accs, color='steelblue')
    ax2.set_title(f'Per-Emotion Accuracy ({best_model})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Accuracy (%)', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(emotion_accs):
        ax2.text(v, i, f' {v:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('./results/final_evaluation_plots.png', dpi=300, bbox_inches='tight')
    print(f"✅ Plots saved to ./results/final_evaluation_plots.png")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review results in ./results/")
    print("2. Update PROJECT_REPORT.md with these numbers")
    print("3. Optionally test ensemble for further improvement")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
