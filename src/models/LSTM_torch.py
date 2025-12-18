import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import argparse
import os
from datetime import datetime
import json
import logging
from pathlib import Path

# GPU тохиргоо
def setup_gpu():
    """
    GPU-г зөв тохируулах ба мэдээлэл харуулах.
    
    Returns:
        device: torch device (cuda эсвэл cpu)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("\n" + "="*80)
        print("GPU МЭДЭЭЛЭЛ")
        print("="*80)
        print(f"✓ GPU олдлоо: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA хувилбар: {torch.version.cuda}")
        print(f"✓ GPU тоо: {torch.cuda.device_count()}")
        print(f"✓ Идэвхтэй GPU: {torch.cuda.current_device()}")
        print(f"✓ GPU санах ой: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("="*80 + "\n")
    else:
        device = torch.device('cpu')
        print("\n⚠ GPU олдсонгүй - CPU ашиглана\n")
    return device

def load_config(config_path='config/lstm_config.json'):
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        config: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config файл олдсонгүй: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def setup_logger(log_file):
    """
    Setup logger with both file and console handlers.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger('LSTM_PyTorch')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_embeddings(embedding_type, config, max_samples=None):
    """
    Load embeddings based on the specified type.
    
    Args:
        embedding_type: Type of embedding to load
        config: Configuration dictionary
        max_samples: Maximum number of samples to use
        
    Returns:
        X: Feature embeddings
        y: Labels
    """
    base_path = config['paths']['embedding_base_path']
    embedding_files = config['embeddings']
    
    embedding_files_full = {
        key: os.path.join(base_path, filename)
        for key, filename in embedding_files.items()
    }
    
    if embedding_type not in embedding_files:
        raise ValueError(f"Embedding type '{embedding_type}' танигдахгүй байна. "
                        f"Боломжтой сонголтууд: {list(embedding_files_full.keys())}")
    
    embedding_path = embedding_files_full[embedding_type]
    
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding файл олдсонгүй: {embedding_path}")
    
    print(f"\nЭмбеддинг ачааллаж байна: {embedding_type}")
    print(f"Файлын зам: {embedding_path}")
    
    X_tensor = torch.load(embedding_path, map_location='cpu')
    
    # Handle 3D shape
    if len(X_tensor.shape) == 3:
        print(f"3D tensor илрүүлсэн: {X_tensor.shape}, mean pooling ашиглаж байна...")
        X = X_tensor.mean(dim=1).numpy()
    else:
        X = X_tensor.numpy()
    
    # Load labels
    y_path = f'{base_path}\\y.pt'
    if os.path.exists(y_path):
        y_tensor = torch.load(y_path, map_location='cpu')
        y = y_tensor.numpy()
    else:
        raise FileNotFoundError(f"Label файл олдсонгүй: {y_path}")
    
    # Limit samples if specified
    if max_samples is not None and max_samples < len(X):
        print(f"Эхний {max_samples} sample ашиглаж байна...")
        X = X[:max_samples]
        y = y[:max_samples]
    
    print(f"Эмбеддингийн хэмжээ: {X.shape}")
    print(f"Лэйбэлийн тоо: {len(y)}")
    
    return X, y

class LSTMClassifier(nn.Module):
    """
    PyTorch LSTM classifier for sentiment analysis.
    """
    def __init__(self, input_dim, lstm_units=128, dropout_rate=0.3, bidirectional=False, num_layers=1):
        super(LSTMClassifier, self).__init__()
        
        self.lstm_units = lstm_units
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dense layers
        lstm_output_dim = lstm_units * 2 if bidirectional else lstm_units
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # Reshape for LSTM: (batch_size, seq_len=1, input_dim)
        x = x.unsqueeze(1)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Dropout
        out = self.dropout(hidden)
        
        # Dense layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        # Move batch to device (GPU/CPU)
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch.float())
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs.squeeze() > 0.5).float()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
        
        # Clear batch from GPU memory
        del X_batch, y_batch, outputs, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, val_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch.float())
            
            total_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            
            # Clear batch from memory
            del X_batch, y_batch, outputs, loss
    
    return total_loss / len(val_loader), correct / total

def predict(model, test_loader, device):
    """Make predictions."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            outputs = model(X_batch)
            predicted = (outputs.squeeze() > 0.5).float()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            
            # Clear batch from memory
            del X_batch, outputs, predicted
    
    return np.array(all_predictions), np.array(all_labels)

def LSTMModel(X, y, embedding_type, config, device):
    """
    Train PyTorch LSTM model and save results to log file.
    
    Args:
        X: Feature embeddings
        y: Labels
        embedding_type: Type of embedding used
        config: Configuration dictionary
        device: torch device
    """
    log_dir = config['paths']['log_dir']
    test_size = config['training']['test_size']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "lstm_pytorch_experiments.log")
    logger = setup_logger(log_file)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        logger.info("\n" + "="*80)
        logger.info(f"НЭВТРЭХ: {embedding_type.upper()} ЭМБЕДДИНГ АШИГЛАН СУРГАЛТ - PyTorch LSTM")
        logger.info("="*80)
        logger.info(f"Огноо ба цаг: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Эмбеддингийн төрөл: {embedding_type}")
        logger.info(f"Өгөгдлийн хэмжээ: {X.shape}")
        logger.info(f"Лэйбэлийн тоо: {len(y)}")
        logger.info(f"Төхөөрөмж: {device}")
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        configs = config['hyperparameters']
        logger.info(f"[{embedding_type}] {len(configs)} конфигурац туршиж байна...")
        
        # Split data
        logger.info(f"[{embedding_type}] Өгөгдлийг train/test хувааж байна (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        logger.info(f"[{embedding_type}] Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

        # Convert to tensors - KEEP ON CPU for memory efficiency
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        logger.info(f"[{embedding_type}] Өгөгдлийг CPU дээр үлдээж, batch-аар GPU-руу илгээнэ (Memory efficient)")
        
        # Create data loaders - data stays on CPU
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True if device.type == 'cuda' else False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True if device.type == 'cuda' else False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True if device.type == 'cuda' else False)
        
        # Clear GPU cache if available
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info(f"[{embedding_type}] GPU санах ойг цэвэрлэлээ")

        best_model = None
        best_val_acc = 0
        best_config = None
        results_list = []
        
        for idx, config_item in enumerate(configs, 1):
            logger.info(f"\n[{embedding_type}] Конфигурац {idx}/{len(configs)}: {config_item}")
            
            start_time = time.time()
            
            # Create model
            model = LSTMClassifier(
                input_dim=X.shape[1],
                lstm_units=config_item['lstm_units'],
                dropout_rate=config_item['dropout'],
                bidirectional=config_item['bidirectional']
            ).to(device)
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=config_item['lr'])
            
            # Early stopping variables
            patience = config['callbacks']['early_stopping']['patience']
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            # Train model
            logger.info(f"[{embedding_type}] Сургалт эхлэв...")
            for epoch in range(epochs):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.debug(f"[{embedding_type}] Early stopping at epoch {epoch+1}")
                    break
            
            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            train_time = time.time() - start_time
            
            # Evaluate on validation set
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            logger.info(f"[{embedding_type}] Config {idx} - Val Accuracy: {val_acc:.4f} | Time: {train_time:.2f}s")
            
            results_list.append({
                'config': config_item,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_time': train_time
            })
            
            # Keep best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save model state dict instead of model reference to save memory
                best_model = model
                best_config = config_item
            
            # Clear GPU cache after each configuration
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                # Log GPU memory usage
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                logger.debug(f"[{embedding_type}] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        logger.info("-"*80)
        logger.info(f"[{embedding_type}] ҮР ДҮН - VALIDATION")
        logger.info("-"*80)
        logger.info(f"[{embedding_type}] Best Val Accuracy: {best_val_acc:.4f}")
        logger.info(f"[{embedding_type}] Best Config: {best_config}")
        
        # Sort and show top configurations
        results_list.sort(key=lambda x: x['val_acc'], reverse=True)
        logger.info(f"[{embedding_type}] Top Configurations:")
        for idx, result in enumerate(results_list[:3], 1):
            logger.info(f"[{embedding_type}]   {idx}. {result['config']} → Val Acc: {result['val_acc']:.4f}")
    
        # Evaluate on test set
        logger.info("-"*80)
        logger.info(f"[{embedding_type}] ҮР ДҮН - TEST SET EVALUATION")
        logger.info("-"*80)
        
        logger.info(f"[{embedding_type}] Test set дээр үнэлж байна...")
        y_pred, y_true = predict(best_model, test_loader, device)
        test_accuracy = accuracy_score(y_true, y_pred)
        
        logger.info(f"[{embedding_type}] Test Accuracy: {test_accuracy:.4f}")
        
        logger.info(f"[{embedding_type}] Classification Report:")
        report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])
        for line in report.split('\n'):
            if line.strip():
                logger.info(f"[{embedding_type}]   {line}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"[{embedding_type}] Confusion Matrix:")
        logger.info(f"[{embedding_type}]   TN={cm[0,0]:5d} | FP={cm[0,1]:5d}")
        logger.info(f"[{embedding_type}]   FN={cm[1,0]:5d} | TP={cm[1,1]:5d}")
    
        # Save results to CSV
        summary_csv = os.path.join(log_dir, "all_experiments_results.csv")
        
        summary_data = {
            'timestamp': timestamp,
            'model': 'LSTM_PyTorch',
            'embedding_type': embedding_type,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'features': X.shape[1],
            'best_val_accuracy': best_val_acc,
            'test_accuracy': test_accuracy,
            'training_time_sec': sum(r['train_time'] for r in results_list),
            'best_lstm_units': best_config['lstm_units'],
            'best_dropout': best_config['dropout'],
            'best_bidirectional': best_config['bidirectional'],
            'best_lr': best_config['lr'],
            'device': str(device),
            'TP': int(cm[1,1]),
            'TN': int(cm[0,0]),
            'FP': int(cm[0,1]),
            'FN': int(cm[1,0])
        }
        
        summary_df = pd.DataFrame([summary_data])
        if os.path.exists(summary_csv):
            existing_df = pd.read_csv(summary_csv)
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"[{embedding_type}] Нэгдсэн үр дүн хадгалагдсан: {summary_csv}")
        
        # Save detailed results
        detailed_csv = os.path.join(log_dir, f"detailed_lstm_pytorch_{embedding_type}_{timestamp}.csv")
        results_df = pd.DataFrame(results_list)
        results_df['model'] = 'LSTM_PyTorch'
        results_df['embedding_type'] = embedding_type
        results_df['timestamp'] = timestamp
        results_df['device'] = str(device)
        results_df.to_csv(detailed_csv, index=False)
        logger.debug(f"[{embedding_type}] Дэлгэрэнгүй үр дүн: {detailed_csv}")
        
        # Save best model
        models_dir = config['paths']['models_dir']
        os.makedirs(models_dir, exist_ok=True)
        model_file = os.path.join(models_dir, f"best_lstm_pytorch_{embedding_type}_{timestamp}.pt")
        torch.save(best_model.state_dict(), model_file)
        logger.info(f"[{embedding_type}] Шилдэг модел: {model_file}")
        
        logger.info(f"[{embedding_type}] АМЖИЛТТАЙ ДУУСЛАА!")
        logger.info("="*80 + "\n")
        
        return best_model
        
    except Exception as e:
        logger.error(f"[{embedding_type}] АЛДАА ГАРЛАА: {str(e)}")
        logger.exception(f"[{embedding_type}] Дэлгэрэнгүй алдаа:")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch LSTM-ээр текстийн сэтгэл хөдлөл таних')
    parser.add_argument(
        '--embedding',
        type=str,
        default='bert_uncased',
        help='Ашиглах эмбеддингийн төрөл (default: bert_uncased)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/lstm_config.json',
        help='Config файлын зам (default: config/lstm_config.json)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Ашиглах дээд sample тоо (config-г давж бичнэ)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (default: config-с уншина, эсвэл 16-г санал болгоно)'
    )
    
    args = parser.parse_args()
    
    # Setup GPU
    device = setup_gpu()
    
    # Load config
    print("Config ачаалж байна...")
    config = load_config(args.config)
    
    # Override max_samples from command line if provided
    max_samples = args.max_samples if args.max_samples else config['training'].get('max_samples')
    
    # Override batch_size from command line if provided
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        print(f"⚠ Batch size-г {args.batch_size} болгож тохируулсан (GPU memory хэмнэх)")
    
    print(f"Config файл: {args.config}")
    print(f"Embedding: {args.embedding}")
    print(f"Max samples: {max_samples}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Device: {device}")
    
    # Warning for GPU memory
    if device.type == 'cuda' and max_samples and max_samples > 10000:
        print(f"\n⚠ АНХААРУУЛГА: {max_samples} sample их байна!")
        print(f"   GPU memory дүүрвэл --batch_size 16 эсвэл 8 ашиглаарай")
        print(f"   Эсвэл --max_samples 5000 гэх мэт багасгаарай\n")
    
    # Load embeddings
    X, y = load_embeddings(args.embedding, config, max_samples=max_samples)
    
    # Train model
    print(f"\nЭмбеддинг: {args.embedding.upper()}")
    model = LSTMModel(X, y, args.embedding, config, device)
    
    print("\n✓ Сургалт амжилттай дууслаа!")
    print(f"✓ Логууд: logs/lstm_pytorch_experiments.log")
    print(f"✓ Үр дүн: logs/all_experiments_results.csv")

# Жишээ:
# python src/models/LSTM_torch.py --embedding bert_uncased
# python src/models/LSTM_torch.py --embedding roberta --max_samples 5000
# python src/models/LSTM_torch.py --embedding albert
