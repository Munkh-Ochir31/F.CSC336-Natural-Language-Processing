"""
BERT Models for Sentiment Analysis
Supports: BERT, RoBERTa, ALBERT, HateBERT, SBERT
"""

import torch
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification
)
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np


class BERTSentimentModel:
    """BERT-based sentiment analysis model"""
    
    def __init__(self, model_type='bert-base-uncased', device=None):
        """
        Initialize BERT model
        
        Args:
            model_type: str - Model name/path
                - 'bert-base-uncased': Standard BERT
                - 'bert-base-cased': Case-sensitive BERT
                - 'roberta-base': RoBERTa
                - 'albert-base-v2': ALBERT
                - 'GroNLP/hateBERT': HateBERT
                - Path to saved model directory
            device: str - 'cuda' or 'cpu'
        """
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self._load_model()
        
    def _load_model(self):
        """Load tokenizer and model based on type"""
        if 'roberta' in self.model_type.lower():
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_type)
            self.model = RobertaForSequenceClassification.from_pretrained(
                self.model_type, num_labels=2
            )
        elif 'albert' in self.model_type.lower():
            self.tokenizer = AlbertTokenizer.from_pretrained(self.model_type)
            self.model = AlbertForSequenceClassification.from_pretrained(
                self.model_type, num_labels=2
            )
        else:  # BERT or HateBERT
            self.tokenizer = BertTokenizer.from_pretrained(self.model_type)
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_type, num_labels=2
            )
        
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, text, max_length=128):
        """
        Predict sentiment for a single text
        
        Args:
            text: str - Input text
            max_length: int - Max sequence length
            
        Returns:
            dict: {
                'label': int (0=negative, 1=positive),
                'sentiment': str ('negative' or 'positive'),
                'confidence': float,
                'probabilities': dict
            }
        """
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        return {
            'label': prediction,
            'sentiment': 'positive' if prediction == 1 else 'negative',
            'confidence': confidence,
            'probabilities': {
                'negative': probs[0][0].item(),
                'positive': probs[0][1].item()
            }
        }
    
    def predict_batch(self, texts, max_length=128, batch_size=16):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: list of str
            max_length: int
            batch_size: int
            
        Returns:
            list of prediction dicts
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                predictions = torch.argmax(probs, dim=1).cpu().numpy()
                
                # Process results
                for j, pred in enumerate(predictions):
                    confidence = probs[j][pred].item()
                    results.append({
                        'label': int(pred),
                        'sentiment': 'positive' if pred == 1 else 'negative',
                        'confidence': confidence,
                        'probabilities': {
                            'negative': probs[j][0].item(),
                            'positive': probs[j][1].item()
                        }
                    })
        
        return results


class SBERTSentimentModel:
    """SBERT-based sentiment analysis model with classifier"""
    
    def __init__(self, model_path='all-MiniLM-L6-v2', classifier_path=None):
        """
        Initialize SBERT model
        
        Args:
            model_path: str - SBERT model name or path
            classifier_path: str - Path to saved classifier (.pkl)
        """
        self.model = SentenceTransformer(model_path)
        self.classifier = None
        
        if classifier_path:
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
    
    def load_classifier(self, path):
        """Load classifier from pickle file"""
        with open(path, 'rb') as f:
            self.classifier = pickle.load(f)
    
    def predict(self, text):
        """
        Predict sentiment for a single text
        
        Args:
            text: str
            
        Returns:
            dict: prediction result
        """
        if self.classifier is None:
            raise ValueError("Classifier not loaded. Use load_classifier() first.")
        
        # Generate embedding
        embedding = self.model.encode([text], convert_to_numpy=True)
        
        # Predict
        prediction = self.classifier.predict(embedding)[0]
        probability = self.classifier.predict_proba(embedding)[0]
        confidence = probability[prediction]
        
        return {
            'label': int(prediction),
            'sentiment': 'positive' if prediction == 1 else 'negative',
            'confidence': float(confidence),
            'probabilities': {
                'negative': float(probability[0]),
                'positive': float(probability[1])
            }
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        if self.classifier is None:
            raise ValueError("Classifier not loaded. Use load_classifier() first.")
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Predict
        predictions = self.classifier.predict(embeddings)
        probabilities = self.classifier.predict_proba(embeddings)
        
        results = []
        for pred, probs in zip(predictions, probabilities):
            results.append({
                'label': int(pred),
                'sentiment': 'positive' if pred == 1 else 'negative',
                'confidence': float(probs[pred]),
                'probabilities': {
                    'negative': float(probs[0]),
                    'positive': float(probs[1])
                }
            })
        
        return results
    
    def get_embeddings(self, texts):
        """Get sentence embeddings"""
        return self.model.encode(texts, convert_to_numpy=True)


def compare_models(text, models_dict):
    """
    Compare predictions from multiple models
    
    Args:
        text: str - Input text
        models_dict: dict - {'model_name': model_object}
        
    Returns:
        dict: Results from all models
    """
    results = {}
    
    for name, model in models_dict.items():
        try:
            prediction = model.predict(text)
            results[name] = prediction
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results


# Example usage
if __name__ == "__main__":
    # Test BERT
    print("Testing BERT models...")
    
    text = "This movie was absolutely fantastic! The acting was superb."
    
    # BERT Base Uncased
    bert_model = BERTSentimentModel('bert-base-uncased')
    result = bert_model.predict(text)
    print(f"\nBERT Base Uncased: {result['sentiment']} (confidence: {result['confidence']:.2%})")
    
    # RoBERTa
    roberta_model = BERTSentimentModel('roberta-base')
    result = roberta_model.predict(text)
    print(f"RoBERTa: {result['sentiment']} (confidence: {result['confidence']:.2%})")
    
    # Compare all models
    models = {
        'BERT': bert_model,
        'RoBERTa': roberta_model
    }
    
    comparison = compare_models(text, models)
    print("\nComparison:")
    for model_name, pred in comparison.items():
        if 'error' not in pred:
            print(f"  {model_name}: {pred['sentiment']} ({pred['confidence']:.2%})")
