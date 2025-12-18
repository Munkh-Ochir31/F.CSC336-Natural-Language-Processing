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

def load_embeddings(embedding_type):
    """
    Load embeddings based on the specified type.
    
    Args:
        embedding_type: Type of embedding to load
        
    Returns:
        X: Feature embeddings
        y: Labels
    """
    embedding_files = {
        'bert_uncased': 'data/data of embedding/bert_embeddings_uncased.npy',
        'bert_cased': 'data/data of embedding/bert_cased_embeddings.npy',
        'roberta': 'data/data of embedding/roberta_embeddings.npy',
        'albert': 'data/data of embedding/albert_embeddings.npy',
        'sbert': 'data/data of embedding/sbert_embeddings.npy',
        'hatebert': 'data/data of embedding/hatebert_embeddings.npy',
        'word2vec': 'models/word2vec/document_embeddings.npy'
    }
    
    if embedding_type not in embedding_files:
        raise ValueError(f"Embedding type '{embedding_type}' танигдахгүй байна. "
                        f"Боломжтой сонголтууд: {list(embedding_files.keys())}")
    
    embedding_path = embedding_files[embedding_type]
    
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding файл олдсонгүй: {embedding_path}")
    
    print(f"\nЭмбеддинг ачааллаж байна: {embedding_type}")
    print(f"Файлын зам: {embedding_path}")
    
    # Load embeddings
    X = np.load(embedding_path)
    
    # Load labels
    if embedding_type == 'word2vec':
        labels_path = 'models/word2vec/labels.npy'
    else:
        labels_path = 'data/cleaned_label.csv'
    
    if labels_path.endswith('.npy'):
        y = np.load(labels_path)
    else:
        df = pd.read_csv(labels_path)
        y = df['sentiment_label'].values
    
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

def LSTMModel(X, y, embedding_type, log_dir='logs', test_size=0.2, epochs=20, batch_size=32):
    """
    Train LSTM model and save results to log file.
    
    Args:
        X: Feature embeddings
        y: Labels
        embedding_type: Type of embedding used
        log_dir: Directory to save log files
        test_size: Proportion of dataset to include in test split
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
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
        
        # Hyperparameters to try
        configs = [
            {'lstm_units': 128, 'dropout': 0.3, 'bidirectional': False, 'lr': 0.001},
            {'lstm_units': 128, 'dropout': 0.5, 'bidirectional': False, 'lr': 0.001},
            {'lstm_units': 256, 'dropout': 0.3, 'bidirectional': False, 'lr': 0.001},
            {'lstm_units': 128, 'dropout': 0.3, 'bidirectional': True, 'lr': 0.001},
        ]
        
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
        
        for idx, config in enumerate(configs, 1):
            logger.info(f"\n[{embedding_type}] Конфигурац {idx}/{len(configs)}: {config}")
            
            start_time = time.time()
            
            # Create model
            model = create_lstm_model(
                input_dim=X.shape[1],
                lstm_units=config['lstm_units'],
                dropout_rate=config['dropout'],
                bidirectional=config['bidirectional']
            )
            
            # Set learning rate
            model.optimizer.learning_rate.assign(config['lr'])
            
            # Callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
            
            # Train model
            logger.info(f"[{embedding_type}] Сургалт эхлэв...")
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
                'config': config,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_time': train_time
            })
            
            # Keep best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                best_config = config

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
        models_dir = "models"
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
        default='bert_uncased',
        choices=['bert_uncased', 'bert_cased', 'roberta', 'albert', 'sbert', 'hatebert', 'word2vec'],
        help='Ашиглах эмбеддингийн төрөл (default: bert_uncased)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Сургалтын эпох тоо (default: 20)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    
    args = parser.parse_args()
    
    # Load embeddings
    X, y = load_embeddings(args.embedding)
    
    # Train model
    print(f"\nЭмбеддинг: {args.embedding.upper()}")
    model = LSTMModel(X, y, args.embedding, epochs=args.epochs, batch_size=args.batch_size)

# Жишээ:
# python src/models/LSTM.py --embedding bert_uncased
# python src/models/LSTM.py --embedding roberta --epochs 30
# python src/models/LSTM.py --embedding albert --batch_size 64
