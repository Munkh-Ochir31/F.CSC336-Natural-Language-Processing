"""
Sentiment prediction script using trained BERT models
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bert_models import BERTSentimentModel, SBERTSentimentModel, compare_models


def predict_sentiment(text, model_type='bert-base-uncased', model_path=None):
    """
    Predict sentiment for a given text
    
    Args:
        text: str - Input text
        model_type: str - Model type or saved model path
        model_path: str - Path to saved model (optional)
    """
    # Use saved model path if provided
    if model_path and os.path.exists(model_path):
        model_type = model_path
    
    # Initialize model
    if 'sbert' in model_type.lower():
        # SBERT requires classifier
        model = SBERTSentimentModel(model_path or 'all-MiniLM-L6-v2')
        classifier_path = os.path.join(os.path.dirname(model_path), 'sbert_classifier.pkl')
        if os.path.exists(classifier_path):
            model.load_classifier(classifier_path)
        else:
            print(f"Warning: Classifier not found at {classifier_path}")
            return None
    else:
        model = BERTSentimentModel(model_type)
    
    # Predict
    result = model.predict(text)
    
    return result


def predict_batch(texts_file, model_type='bert-base-uncased', output_file=None):
    """
    Predict sentiment for multiple texts from a file
    
    Args:
        texts_file: str - Path to file with texts (one per line)
        model_type: str - Model type
        output_file: str - Output file path (optional)
    """
    # Read texts
    with open(texts_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    # Initialize model
    model = BERTSentimentModel(model_type)
    
    # Predict
    results = model.predict_batch(texts)
    
    # Output
    output_lines = []
    for i, (text, result) in enumerate(zip(texts, results), 1):
        line = f"{i}. {text[:50]}... -> {result['sentiment']} ({result['confidence']:.2%})"
        output_lines.append(line)
        print(line)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"\nResults saved to: {output_file}")
    
    return results


def compare_all_models(text):
    """
    Compare predictions from all 6 BERT models
    
    Args:
        text: str - Input text
    """
    print("="*80)
    print("Comparing all BERT models...")
    print("="*80)
    print(f"Text: {text}\n")
    
    # Define models
    model_configs = {
        'BERT Base Uncased': 'bert-base-uncased',
        'BERT Base Cased': 'bert-base-cased',
        'RoBERTa': 'roberta-base',
        'ALBERT': 'albert-base-v2',
        'HateBERT': 'GroNLP/hateBERT'
    }
    
    # Load models and predict
    results = {}
    for name, model_type in model_configs.items():
        try:
            print(f"Loading {name}...")
            model = BERTSentimentModel(model_type)
            result = model.predict(text)
            results[name] = result
            print(f"  ✓ {result['sentiment']} (confidence: {result['confidence']:.2%})")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[name] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    for name, result in results.items():
        if 'error' not in result:
            emoji = "🟢" if result['sentiment'] == 'positive' else "🔴"
            print(f"{emoji} {name:20s}: {result['sentiment']:10s} ({result['confidence']:.2%})")
        else:
            print(f"❌ {name:20s}: Error")
    
    return results


def interactive_mode():
    """Interactive prediction mode"""
    print("="*80)
    print("BERT Sentiment Analysis - Interactive Mode")
    print("="*80)
    print("\nAvailable models:")
    print("  1. BERT Base Uncased")
    print("  2. BERT Base Cased")
    print("  3. RoBERTa")
    print("  4. ALBERT")
    print("  5. HateBERT")
    print("  6. Compare All")
    print("\nType 'quit' to exit\n")
    
    model_map = {
        '1': 'bert-base-uncased',
        '2': 'bert-base-cased',
        '3': 'roberta-base',
        '4': 'albert-base-v2',
        '5': 'GroNLP/hateBERT'
    }
    
    while True:
        try:
            # Get model choice
            choice = input("\nSelect model (1-6): ").strip()
            if choice.lower() == 'quit':
                break
            
            # Get text
            text = input("Enter text: ").strip()
            if not text or text.lower() == 'quit':
                break
            
            print()
            
            # Predict
            if choice == '6':
                compare_all_models(text)
            elif choice in model_map:
                model = BERTSentimentModel(model_map[choice])
                result = model.predict(text)
                
                emoji = "🟢" if result['sentiment'] == 'positive' else "🔴"
                print(f"{emoji} Prediction: {result['sentiment'].upper()}")
                print(f"   Confidence: {result['confidence']:.2%}")
                print(f"   Probabilities:")
                print(f"     - Negative: {result['probabilities']['negative']:.2%}")
                print(f"     - Positive: {result['probabilities']['positive']:.2%}")
            else:
                print("Invalid choice!")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='BERT Sentiment Analysis Prediction')
    parser.add_argument('--text', '-t', type=str, help='Text to analyze')
    parser.add_argument('--file', '-f', type=str, help='File with texts (one per line)')
    parser.add_argument('--model', '-m', type=str, default='bert-base-uncased',
                        help='Model type (bert-base-uncased, roberta-base, etc.)')
    parser.add_argument('--model-path', '-p', type=str, help='Path to saved model')
    parser.add_argument('--output', '-o', type=str, help='Output file for batch predictions')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare all models')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
    
    # Compare all models
    elif args.compare and args.text:
        compare_all_models(args.text)
    
    # Single text prediction
    elif args.text:
        result = predict_sentiment(args.text, args.model, args.model_path)
        if result:
            emoji = "🟢" if result['sentiment'] == 'positive' else "🔴"
            print(f"\n{emoji} Sentiment: {result['sentiment'].upper()}")
            print(f"   Confidence: {result['confidence']:.2%}")
    
    # Batch prediction
    elif args.file:
        predict_batch(args.file, args.model, args.output)
    
    # Default: show usage
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
