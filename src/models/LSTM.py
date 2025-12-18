import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import argparse
import os
from datetime import datetime
import json
import logging
from pathlib import Path

# GPU тохиргоо
def setup_gpu():
    """
    GPU-г зөв тохируулах - memory growth, device visibility гэх мэт.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory growth-г идэвхжүүлэх (хэрэгцээтэй хэмжээгээр л эзэлнэ)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ {len(gpus)} GPU олдлоо:")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            print(f"✓ GPU memory growth идэвхтэй")
            return True
        except RuntimeError as e:
            print(f"⚠ GPU тохиргоонд алдаа: {e}")
            return False
    else:
        print("⚠ GPU олдсонгүй - CPU ашиглана")
        return False

# GPU тохиргоо хийх
setup_gpu()

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
    # Create logger
    logger = logging.getLogger('LSTM')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simple)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_embeddings(embedding_type, config, max_samples=None):
    """
    Load embeddings based on the specified type.
    
    Args:
        embedding_type: Type of embedding to load
        config: Configuration dictionary
        max_samples: Maximum number of samples to use (default: None = all)
        
    Returns:
        X: Feature embeddings
        y: Labels
    """
    import torch
    
    # Get paths from config
    base_path = config['paths']['embedding_base_path']
    embedding_files = config['embeddings']
    
    # Build full paths
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
    
    # Load embeddings from .pt file
    X_tensor = torch.load(embedding_path, map_location='cpu')
    
    # Convert tensor to numpy and handle 3D shape
    if len(X_tensor.shape) == 3:
        # Shape: (n_samples, seq_len, embedding_dim)
        # Use mean pooling across sequence length
        print(f"3D tensor илрүүлсэн: {X_tensor.shape}, mean pooling ашиглаж байна...")
        X = X_tensor.mean(dim=1).numpy()  # (n_samples, embedding_dim)
    else:
        X = X_tensor.numpy()
    
    # Load labels from y.pt
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

def create_lstm_model(input_dim, lstm_units=128, dropout_rate=0.3, bidirectional=False):
    """
    Create LSTM model architecture.
    
    Args:
        input_dim: Input dimension (embedding size)
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate
        bidirectional: Use bidirectional LSTM
        
    Returns:
        model: Compiled Keras model
    """
    model = Sequential()
    
    # Reshape input for LSTM (add time dimension)
    model.add(tf.keras.layers.Reshape((1, input_dim), input_shape=(input_dim,)))
    
    if bidirectional:
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=False)))
    else:
        model.add(LSTM(lstm_units, return_sequences=False))
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def LSTMModel(X, y, embedding_type, config):
    """
    Train LSTM model and save results to log file.
    
    Args:
        X: Feature embeddings
        y: Labels
        embedding_type: Type of embedding used
        config: Configuration dictionary
    """
    # Check GPU availability
    print("\n" + "="*80)
    print("ТӨХӨӨРӨМЖИЙН МЭДЭЭЛЭЛ")
    print("="*80)
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    if gpu_available:
        print("✓ GPU дээр сургалт явуулна")
        print(f"  TensorFlow GPU хувилбар: {tf.test.is_built_with_cuda()}")
        print(f"  Боломжтой GPU-ийн тоо: {len(tf.config.list_physical_devices('GPU'))}")
    else:
        print("⚠ CPU дээр сургалт явуулна (GPU олдсонгүй)")
    print("="*80 + "\n")
    
    # Get parameters from config
    log_dir = config['paths']['log_dir']
    test_size = config['training']['test_size']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Single log file for all embeddings
    log_file = os.path.join(log_dir, "lstm_experiments.log")
    
    # Setup logger
    logger = setup_logger(log_file)
    
    # Create timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        logger.info("\n" + "="*80)
        logger.info(f"НЭВТРЭХ: {embedding_type.upper()} ЭМБЕДДИНГ АШИГЛАН СУРГАЛТ - LSTM")
        logger.info("="*80)
        logger.info(f"Огноо ба цаг: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Эмбеддингийн төрөл: {embedding_type}")
        logger.info(f"Өгөгдлийн хэмжээ: {X.shape}")
        logger.info(f"Лэйбэлийн тоо: {len(y)}")
        logger.info(f"Төхөөрөмж: {'GPU' if gpu_available else 'CPU'}")
        if gpu_available:
            logger.info(f"GPU-ийн тоо: {len(tf.config.list_physical_devices('GPU'))}")
        
        # Get hyperparameters from config
        configs = config['hyperparameters']
        
        logger.info(f"[{embedding_type}] {len(configs)} конфигурац туршиж байна...")
        
        # Split data into train and test sets
        logger.info(f"[{embedding_type}] Өгөгдлийг train/test хувааж байна (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        # Further split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        logger.info(f"[{embedding_type}] Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

        best_model = None
        best_val_acc = 0
        best_config = None
        results_list = []
        
        for idx, config_item in enumerate(configs, 1):
            logger.info(f"\n[{embedding_type}] Конфигурац {idx}/{len(configs)}: {config_item}")
            
            start_time = time.time()
            
            # Create model
            model = create_lstm_model(
                input_dim=X.shape[1],
                lstm_units=config_item['lstm_units'],
                dropout_rate=config_item['dropout'],
                bidirectional=config_item['bidirectional']
            )
            
            # Set learning rate
            model.optimizer.learning_rate.assign(config_item['lr'])
            
            # Callbacks from config
            early_stop_cfg = config['callbacks']['early_stopping']
            reduce_lr_cfg = config['callbacks']['reduce_lr']
            
            early_stop = EarlyStopping(
                monitor=early_stop_cfg['monitor'],
                patience=early_stop_cfg['patience'],
                restore_best_weights=early_stop_cfg['restore_best_weights']
            )
            reduce_lr = ReduceLROnPlateau(
                monitor=reduce_lr_cfg['monitor'],
                factor=reduce_lr_cfg['factor'],
                patience=reduce_lr_cfg['patience'],
                min_lr=reduce_lr_cfg['min_lr']
            )
            
            # Train model with GPU utilization
            logger.info(f"[{embedding_type}] Сургалт эхлэв...")
            with tf.device('/GPU:0' if gpu_available else '/CPU:0'):
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stop, reduce_lr],
                    verbose=0
                )
            
            train_time = time.time() - start_time
            
            # Evaluate on validation set
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            
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
                best_model = model
                best_config = config_item

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
        y_pred_proba = best_model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        test_accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"[{embedding_type}] Test Accuracy: {test_accuracy:.4f}")
        
        logger.info(f"[{embedding_type}] Classification Report:")
        report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
        for line in report.split('\n'):
            if line.strip():
                logger.info(f"[{embedding_type}]   {line}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"[{embedding_type}] Confusion Matrix:")
        logger.info(f"[{embedding_type}]   TN={cm[0,0]:5d} | FP={cm[0,1]:5d}")
        logger.info(f"[{embedding_type}]   FN={cm[1,0]:5d} | TP={cm[1,1]:5d}")
    
        # Save results to a consolidated CSV
        summary_csv = os.path.join(log_dir, "all_experiments_results.csv")
        
        # Create summary dictionary for this experiment
        summary_data = {
            'timestamp': timestamp,
            'model': 'LSTM',
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
            'TP': int(cm[1,1]),
            'TN': int(cm[0,0]),
            'FP': int(cm[0,1]),
            'FN': int(cm[1,0])
        }
        
        # Append to CSV (create if doesn't exist)
        summary_df = pd.DataFrame([summary_data])
        if os.path.exists(summary_csv):
            existing_df = pd.read_csv(summary_csv)
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"[{embedding_type}] Нэгдсэн үр дүн хадгалагдсан: {summary_csv}")
        
        # Save detailed results
        detailed_csv = os.path.join(log_dir, f"detailed_lstm_{embedding_type}_{timestamp}.csv")
        results_df = pd.DataFrame(results_list)
        results_df['model'] = 'LSTM'
        results_df['embedding_type'] = embedding_type
        results_df['timestamp'] = timestamp
        results_df.to_csv(detailed_csv, index=False)
        logger.debug(f"[{embedding_type}] Дэлгэрэнгүй үр дүн: {detailed_csv}")
        
        # Save best model to models directory
        models_dir = config['paths']['models_dir']
        os.makedirs(models_dir, exist_ok=True)
        model_file = os.path.join(models_dir, f"best_lstm_{embedding_type}_{timestamp}.h5")
        best_model.save(model_file)
        logger.info(f"[{embedding_type}] Шилдэг модел: {model_file}")
        
        logger.info(f"[{embedding_type}] АМЖИЛТТАЙ ДУУСЛАА!")
        logger.info("="*80 + "\n")
        
        return best_model
        
    except Exception as e:
        logger.error(f"[{embedding_type}] АЛДАА ГАРЛАА: {str(e)}")
        logger.exception(f"[{embedding_type}] Дэлгэрэнгүй алдаа:")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM-ээр текстийн сэтгэл хөдлөл таних')
    parser.add_argument(
        '--embedding',
        type=str,
        default='bert_cased',
        help='Ашиглах эмбеддингийн төрөл (default: bert_cased)'
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
    
    args = parser.parse_args()
    
    # Load config
    print("Config ачаалж байна...")
    config = load_config(args.config)
    
    # Override max_samples from command line if provided
    max_samples = args.max_samples if args.max_samples else config['training'].get('max_samples')
    
    print(f"Config файл: {args.config}")
    print(f"Embedding: {args.embedding}")
    print(f"Max samples: {max_samples}")
    
    # Load embeddings
    X, y = load_embeddings(args.embedding, config, max_samples=max_samples)
    
    # Train model
    print(f"\nЭмбеддинг: {args.embedding.upper()}")
    model = LSTMModel(X, y, args.embedding, config)

# Жишээ:
# python src/models/LSTM.py --embedding bert_uncased
# python src/models/LSTM.py --embedding roberta --epochs 30
# python src/models/LSTM.py --embedding albert --batch_size 64
