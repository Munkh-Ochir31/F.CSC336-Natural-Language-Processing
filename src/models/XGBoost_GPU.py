import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier
import time
import pickle
import argparse
import os
from datetime import datetime
import json
import logging

def setup_gpu_info():
    """
    GPU мэдээлэл харуулах ба GPU байгаа эсэхийг шалгах.
    
    Returns:
        gpu_available: GPU байгаа эсэх
    """
    print("\n" + "="*80)
    print("ТӨХӨӨРӨМЖИЙН МЭДЭЭЛЭЛ")
    print("="*80)
    
    # Check PyTorch CUDA
    if torch.cuda.is_available():
        print(f"✓ PyTorch CUDA: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA хувилбар: {torch.version.cuda}")
        print(f"  GPU санах ой: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        gpu_available = True
    else:
        print("⚠ PyTorch CUDA олдсонгүй")
        gpu_available = False
    
    # Check XGBoost GPU support
    print(f"✓ XGBoost хувилбар: {xgb.__version__}")
    try:
        # Try to build a simple XGBoost model with GPU
        test_model = XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
        test_X = np.random.rand(10, 5)
        test_y = np.random.randint(0, 2, 10)
        test_model.fit(test_X, test_y, verbose=False)
        print("✓ XGBoost GPU дэмжлэг идэвхтэй")
        gpu_available = True
    except Exception as e:
        print(f"⚠ XGBoost GPU дэмжлэг идэвхгүй: {str(e)}")
        print("  CPU ашиглана")
        gpu_available = False
    
    print("="*80 + "\n")
    return gpu_available

def setup_logger(log_file):
    """
    Setup logger with both file and console handlers.
    """
    logger = logging.getLogger('XGBoost_GPU')
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

def load_embeddings(embedding_type, config):
    """
    Load embeddings based on the specified type.
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
    
    # Load embeddings from .pt file
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
    
    print(f"Эмбеддингийн хэмжээ: {X.shape}")
    print(f"Лэйбэлийн тоо: {len(y)}")
    
    return X, y

def load_config(config_path='config/lstm_config.json'):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config файл олдсонгүй: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def XGBoostGPU(X, y, embedding_type, config, use_gpu=True):
    """
    Train XGBoost with GPU and save results to log file.
    
    Args:
        X: Feature embeddings
        y: Labels
        embedding_type: Type of embedding used
        config: Configuration dictionary
        use_gpu: Whether to use GPU
    """
    log_dir = config['paths']['log_dir']
    test_size = config['training']['test_size']
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "xgboost_gpu_experiments.log")
    logger = setup_logger(log_file)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        logger.info("\n" + "="*80)
        logger.info(f"НЭВТРЭХ: {embedding_type.upper()} ЭМБЕДДИНГ АШИГЛАН СУРГАЛТ - XGBoost GPU")
        logger.info("="*80)
        logger.info(f"Огноо ба цаг: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Эмбеддингийн төрөл: {embedding_type}")
        logger.info(f"Өгөгдлийн хэмжээ: {X.shape}")
        logger.info(f"Лэйбэлийн тоо: {len(y)}")
        logger.info(f"Төхөөрөмж: {'GPU' if use_gpu else 'CPU'}")
        
        # Hyperparameter grid for XGBoost
        param_grid = [
            # Fast configurations
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8},
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8},
            {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.9},
            {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.9},
            # Medium configurations
            {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8},
            {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.9},
            # Deep configurations
            {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.01, 'subsample': 0.9},
            {'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.01, 'subsample': 0.8},
        ]
        
        logger.info(f"[{embedding_type}] {len(param_grid)} конфигурац туршиж байна...")
        
        # Split data
        logger.info(f"[{embedding_type}] Өгөгдлийг train/test хувааж байна (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        logger.info(f"[{embedding_type}] Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

        best_model = None
        best_val_acc = 0
        best_config = None
        results_list = []
        
        for idx, params in enumerate(param_grid, 1):
            logger.info(f"\n[{embedding_type}] Конфигурац {idx}/{len(param_grid)}: {params}")
            
            start_time = time.time()
            
            # Create XGBoost model
            model = XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                tree_method='hist',  # GPU-optimized histogram method
                device='cuda' if use_gpu else 'cpu',
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            # Train with early stopping
            logger.info(f"[{embedding_type}] Сургалт эхлэв...")
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            train_time = time.time() - start_time
            
            # Evaluate on validation set
            y_val_pred = model.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)
            
            logger.info(f"[{embedding_type}] Config {idx} - Val Accuracy: {val_acc:.4f} | Time: {train_time:.2f}s")
            
            results_list.append({
                'config': params,
                'val_acc': val_acc,
                'train_time': train_time
            })
            
            # Keep best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                best_config = params
            
            # GPU memory cleanup
            if use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        y_pred = best_model.predict(X_test)
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
    
        # Save results to CSV
        summary_csv = os.path.join(log_dir, "all_experiments_results.csv")
        
        summary_data = {
            'timestamp': timestamp,
            'model': 'XGBoost_GPU' if use_gpu else 'XGBoost_CPU',
            'embedding_type': embedding_type,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'features': X.shape[1],
            'best_val_accuracy': best_val_acc,
            'test_accuracy': test_accuracy,
            'training_time_sec': sum(r['train_time'] for r in results_list),
            'best_n_estimators': best_config['n_estimators'],
            'best_max_depth': best_config['max_depth'],
            'best_learning_rate': best_config['learning_rate'],
            'best_subsample': best_config['subsample'],
            'device': 'GPU' if use_gpu else 'CPU',
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
        detailed_csv = os.path.join(log_dir, f"detailed_xgboost_gpu_{embedding_type}_{timestamp}.csv")
        results_df = pd.DataFrame(results_list)
        results_df['model'] = 'XGBoost_GPU' if use_gpu else 'XGBoost_CPU'
        results_df['embedding_type'] = embedding_type
        results_df['timestamp'] = timestamp
        results_df.to_csv(detailed_csv, index=False)
        logger.debug(f"[{embedding_type}] Дэлгэрэнгүй үр дүн: {detailed_csv}")
        
        # Save best model
        models_dir = config['paths']['models_dir']
        os.makedirs(models_dir, exist_ok=True)
        model_file = os.path.join(models_dir, f"best_xgboost_gpu_{embedding_type}_{timestamp}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(best_model, f)
        logger.info(f"[{embedding_type}] Шилдэг модел: {model_file}")
        
        logger.info(f"[{embedding_type}] АМЖИЛТТАЙ ДУУСЛАА!")
        logger.info("="*80 + "\n")
        
        return best_model
        
    except Exception as e:
        logger.error(f"[{embedding_type}] АЛДАА ГАРЛАА: {str(e)}")
        logger.exception(f"[{embedding_type}] Дэлгэрэнгүй алдаа:")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XGBoost GPU-ээр текстийн сэтгэл хөдлөл таних')
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
        '--no_gpu',
        action='store_true',
        help='GPU ашиглахгүй байх (CPU mode)'
    )
    
    args = parser.parse_args()
    
    # Check GPU availability
    gpu_available = setup_gpu_info()
    use_gpu = gpu_available and not args.no_gpu
    
    if args.no_gpu:
        print("⚠ --no_gpu flag-аар CPU mode ашиглана\n")
    
    # Load config
    print("Config ачаалж байна...")
    config = load_config(args.config)
    
    print(f"Config файл: {args.config}")
    print(f"Embedding: {args.embedding}")
    print(f"Device: {'GPU' if use_gpu else 'CPU'}")
    
    # Load embeddings
    X, y = load_embeddings(args.embedding, config)
    
    # Train model
    print(f"\nЭмбеддинг: {args.embedding.upper()}")
    model = XGBoostGPU(X, y, args.embedding, config, use_gpu=use_gpu)
    
    print("\n✓ Сургалт амжилттай дууслаа!")
    print(f"✓ Логууд: logs/xgboost_gpu_experiments.log")
    print(f"✓ Үр дүн: logs/all_experiments_results.csv")

# Жишээ:
# python src/models/XGBoost_GPU.py --embedding bert_uncased
# python src/models/XGBoost_GPU.py --embedding roberta
# python src/models/XGBoost_GPU.py --embedding albert --no_gpu  # CPU mode
