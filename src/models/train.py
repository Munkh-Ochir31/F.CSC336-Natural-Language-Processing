"""
Training script for BERT models
"""

import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import time


class SentimentDataset(Dataset):
    """Dataset for sentiment analysis"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_tokenizer_and_model(model_type):
    """Load tokenizer and model based on type"""
    if 'roberta' in model_type.lower():
        tokenizer = RobertaTokenizer.from_pretrained(model_type)
        model = RobertaForSequenceClassification.from_pretrained(model_type, num_labels=2)
    elif 'albert' in model_type.lower():
        tokenizer = AlbertTokenizer.from_pretrained(model_type)
        model = AlbertForSequenceClassification.from_pretrained(model_type, num_labels=2)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_type)
        model = BertForSequenceClassification.from_pretrained(model_type, num_labels=2)
    
    return tokenizer, model


def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    losses = []
    correct_predictions = 0
    
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': sum(losses) / len(losses)})
    
    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(losses)


def eval_model(model, data_loader, device):
    """Evaluate model"""
    model.eval()
    losses = []
    correct_predictions = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return (correct_predictions.double() / len(data_loader.dataset), 
            sum(losses) / len(losses), 
            predictions, 
            true_labels)


def train_model(
    data_path,
    model_type='bert-base-uncased',
    output_dir='./model_output',
    epochs=1,
    batch_size=16,
    learning_rate=2e-5,
    max_length=128,
    test_size=0.2
):
    """
    Train a BERT model for sentiment analysis
    
    Args:
        data_path: str - Path to CSV file with columns: review_text, sentiment_label
        model_type: str - Model name (bert-base-uncased, roberta-base, etc.)
        output_dir: str - Directory to save trained model
        epochs: int - Number of training epochs
        batch_size: int - Batch size
        learning_rate: float - Learning rate
        max_length: int - Max sequence length
        test_size: float - Test set ratio
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['review_text'].values,
        df['sentiment_label'].values,
        test_size=test_size,
        random_state=42,
        stratify=df['sentiment_label'].values
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Load tokenizer and model
    print(f"\nLoading model: {model_type}")
    tokenizer, model = load_tokenizer_and_model(model_type)
    model = model.to(device)
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, max_length)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    start_time = time.time()
    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 80)
        
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}")
        
        val_acc, val_loss, predictions, true_labels = eval_model(model, test_loader, device)
        print(f"Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.4f}")
        
        history['train_acc'].append(train_acc.item())
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc.item())
        history['val_loss'].append(val_loss)
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Test Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['Negative', 'Positive']))
    
    # Save model
    import os
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to: {output_dir}")
    
    return model, tokenizer, history


def main():
    parser = argparse.ArgumentParser(description='Train BERT model for sentiment analysis')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                        help='Model type (bert-base-uncased, roberta-base, albert-base-v2, etc.)')
    parser.add_argument('--output', type=str, default='./model_output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=128, help='Max sequence length')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        model_type=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()
