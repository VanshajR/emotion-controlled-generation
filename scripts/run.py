"""
Utility script for common project tasks.

Usage:
    python scripts/run.py test          # Run tests
    python scripts/run.py setup         # Run setup verification
    python scripts/run.py clean         # Clean temporary files
    python scripts/run.py info          # Display project info
"""

import os
import sys
import shutil
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


def run_tests():
    """Run all tests."""
    print("Running tests...\n")
    
    test_files = [
        'tests/test_emotion_mapping.py'
    ]
    
    for test_file in test_files:
        test_path = os.path.join(PROJECT_ROOT, test_file)
        if os.path.exists(test_path):
            print(f"Running {test_file}...")
            os.system(f'python "{test_path}"')
            print()
        else:
            print(f"⚠️  Test file not found: {test_file}")


def run_setup():
    """Run setup verification."""
    print("Running setup verification...\n")
    setup_path = os.path.join(PROJECT_ROOT, 'setup.py')
    if os.path.exists(setup_path):
        os.system(f'python "{setup_path}"')
    else:
        print("⚠️  setup.py not found")


def clean_project():
    """Clean temporary files and caches."""
    print("Cleaning project...\n")
    
    patterns_to_remove = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.Python',
        '*.so',
        '.ipynb_checkpoints',
        '.DS_Store'
    ]
    
    removed_count = 0
    
    # Walk through project directory
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            cache_dir = os.path.join(root, '__pycache__')
            print(f"Removing {cache_dir}")
            shutil.rmtree(cache_dir)
            removed_count += 1
        
        # Remove .ipynb_checkpoints
        if '.ipynb_checkpoints' in dirs:
            checkpoint_dir = os.path.join(root, '.ipynb_checkpoints')
            print(f"Removing {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)
            removed_count += 1
        
        # Remove .pyc files
        for file in files:
            if file.endswith('.pyc') or file.endswith('.pyo') or file.endswith('.pyd'):
                file_path = os.path.join(root, file)
                print(f"Removing {file_path}")
                os.remove(file_path)
                removed_count += 1
    
    print(f"\n✅ Cleaned {removed_count} items")


def display_info():
    """Display project information."""
    print("="*70)
    print("EMOTION-CONTROLLED TEXT GENERATION - PROJECT INFO")
    print("="*70)
    
    # Project structure
    print("\n📁 Project Structure:")
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        level = root.replace(PROJECT_ROOT, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        
        if level < 2:  # Only show files in top 2 levels
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                if not file.startswith('.') and not file.endswith('.pyc'):
                    print(f'{subindent}{file}')
    
    # Count files
    print("\n📊 File Counts:")
    notebooks = len([f for f in os.listdir(os.path.join(PROJECT_ROOT, 'notebooks')) if f.endswith('.ipynb')])
    utils = len([f for f in os.listdir(os.path.join(PROJECT_ROOT, 'utils')) if f.endswith('.py')])
    
    print(f"  Notebooks: {notebooks}")
    print(f"  Utility modules: {utils}")
    
    # Check models
    print("\n🤖 Model Status:")
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    models = ['emotion_classifier_roberta', 'gpt2_baseline', 'gpt2_prefix', 'gpt2_tokens', 'gpt2_lora']
    
    for model in models:
        model_path = os.path.join(models_dir, model)
        if os.path.exists(model_path) and os.listdir(model_path):
            print(f"  ✅ {model}")
        else:
            print(f"  ❌ {model} (not trained)")
    
    # Configuration
    print("\n⚙️  Configuration:")
    try:
        from config import EMOTIONS
        print(f"  Target emotions: {', '.join(EMOTIONS['target'])}")
    except:
        print("  ⚠️  Could not load config.py")
    
    print("\n" + "="*70)


def list_notebooks():
    """List all available notebooks."""
    print("📓 Available Notebooks:\n")
    
    notebooks_dir = os.path.join(PROJECT_ROOT, 'notebooks')
    notebooks = sorted([f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')])
    
    descriptions = {
        '00_quick_start.ipynb': 'Quick demonstration of emotion classification and generation',
        '01_train_emotion_classifier.ipynb': 'Fine-tune RoBERTa on GoEmotions (Part 1)',
        '02_train_gpt2_variants.ipynb': 'Train GPT-2 with 4 conditioning methods (Part 2)',
        '03_evaluation.ipynb': 'Comprehensive evaluation and ablation study (Part 3)'
    }
    
    for i, notebook in enumerate(notebooks, 1):
        desc = descriptions.get(notebook, 'No description')
        print(f"{i}. {notebook}")
        print(f"   {desc}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Utility script for Emotion-Controlled Text Generation project'
    )
    
    parser.add_argument(
        'command',
        choices=['test', 'setup', 'clean', 'info', 'notebooks'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    commands = {
        'test': run_tests,
        'setup': run_setup,
        'clean': clean_project,
        'info': display_info,
        'notebooks': list_notebooks
    }
    
    commands[args.command]()


if __name__ == "__main__":
    main()
