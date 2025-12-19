import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import pickle
import argparse
import os
from datetime import datetime
import json
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import load_config, get_path, resolve_path, ensure_dir

def setup_logger(log_file):
    """
    Setup logger with both file and console handlers.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('LogisticRegression')
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

def load_embeddings(embedding_type, paths_config):
    """
    Load embeddings based on the specified type.
    
    Args:
        embedding_type: Type of embedding to load
        paths_config: Loaded paths configuration
        
    Returns:
        X: Feature embeddings
        y: Labels
    """
    # Get embedding path from config
    try:
        if embedding_type == 'word2vec':
            embedding_path = resolve_path(get_path(paths_config, 'embeddings', 'word2vec_embeddings'))
        else:
            embedding_path = resolve_path(get_path(paths_config, 'embeddings', embedding_type))
    except KeyError:
        raise ValueError(f"Embedding type '{embedding_type}' танигдахгүй байна config файлд")
    
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding файл олдсонгүй: {embedding_path}")
    
    print(f"\nЭмбеддинг ачааллаж байна: {embedding_type}")
    print(f"Файлын зам: {embedding_path}")
    
    # Load embeddings
    X = np.load(embedding_path)
    
    # Load labels
    if embedding_type == 'word2vec':
        labels_path = resolve_path(get_path(paths_config, 'embeddings', 'word2vec_labels'))
    else:
        labels_path = resolve_path(get_path(paths_config, 'data', 'cleaned_label'))
    
    if labels_path.endswith('.npy'):
        y = np.load(labels_path)
    else:
        df = pd.read_csv(labels_path)
        y = df['sentiment_label'].values
    
    print(f"Эмбеддингийн хэмжээ: {X.shape}")
    print(f"Лэйбэлийн тоо: {len(y)}")
    
    return X, y

def LogisticRegress(X, y, embedding_type, lr_config, paths_config):
    """
    Train Logistic Regression with Grid Search and save results to log file.
    
    Args:
        X: Feature embeddings
        y: Labels
        embedding_type: Type of embedding used
        lr_config: Loaded Logistic Regression configuration
        paths_config: Loaded paths configuration
    """
    from sklearn.model_selection import train_test_split
    
    # Get parameters from config
    test_size = lr_config.get('test_size', 0.2)
    random_state = lr_config.get('random_state', 42)
    cv_folds = lr_config.get('cv_folds', 5)
    param_grid = lr_config.get('param_grid', {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [500, 1000]
    })
    
    # Get output paths from config
    log_dir = resolve_path(get_path(paths_config, 'output', 'logs'))
    models_dir = resolve_path(get_path(paths_config, 'output', 'models'))
    
    # Create directories
    ensure_dir(log_dir)
    ensure_dir(models_dir)
    
    # Single log file for all embeddings
    log_file = os.path.join(log_dir, "logistic_regression_experiments.log")
    
    # Setup logger
    logger = setup_logger(log_file)
    
    # Create timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        logger.info("\n" + "="*80)
        logger.info(f"НЭВТРЭХ: {embedding_type.upper()} ЭМБЕДДИНГ АШИГЛАН СУРГАЛТ")
        logger.info("="*80)
        logger.info(f"Огноо ба цаг: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Эмбеддингийн төрөл: {embedding_type}")
        logger.info(f"Өгөгдлийн хэмжээ: {X.shape}")
        logger.info(f"Лэйбэлийн тоо: {len(y)}")
        
        logger.debug(f"[{embedding_type}] Параметрүүдийн Grid:")
        logger.debug(json.dumps(param_grid, indent=2))
        
        # Split data into train and test sets
        logger.info(f"[{embedding_type}] Өгөгдлийг train/test хувааж байна (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"[{embedding_type}] Train: {X_train.shape} | Test: {X_test.shape}")

        start_time = time.time()

        # Create model
        logger.info(f"[{embedding_type}] Logistic Regression модел үүсгэж байна...")
        lr_model = LogisticRegression(random_state=random_state)

        # Grid search with cross-validation
        logger.info(f"[{embedding_type}] Grid Search with {cv_folds}-Fold CV эхлүүлж байна...")
        lr_grid = GridSearchCV(
            estimator=lr_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )

        logger.info(f"[{embedding_type}] Сургалт эхлэв...")
        lr_grid.fit(X_train, y_train)

        lr_time = time.time() - start_time

        logger.info("-"*80)
        logger.info(f"[{embedding_type}] ҮР ДҮН - CROSS-VALIDATION")
        logger.info("-"*80)
        logger.info(f"[{embedding_type}] Best CV Score: {lr_grid.best_score_:.4f}")
        logger.info(f"[{embedding_type}] Best Parameters: {lr_grid.best_params_}")
        logger.info(f"[{embedding_type}] Training Time: {lr_time:.2f} sec ({lr_time/60:.2f} min)")

        # Get top 5 results
        results_df = pd.DataFrame(lr_grid.cv_results_)
        top_5 = results_df.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
        logger.info(f"[{embedding_type}] Top 5 Configurations:")
        for idx, row in top_5.iterrows():
            logger.info(f"[{embedding_type}]   {row['params']} → Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
    
        # Evaluate on test set
        logger.info("-"*80)
        logger.info(f"[{embedding_type}] ҮР ДҮН - TEST SET EVALUATION")
        logger.info("-"*80)
        
        logger.info(f"[{embedding_type}] Test set дээр үнэлж байна...")
        y_pred = lr_grid.best_estimator_.predict(X_test)
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
            'embedding_type': embedding_type,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'features': X.shape[1],
            'best_cv_score': lr_grid.best_score_,
            'test_accuracy': test_accuracy,
            'training_time_sec': lr_time,
            'best_C': lr_grid.best_params_['C'],
            'best_solver': lr_grid.best_params_['solver'],
            'best_max_iter': lr_grid.best_params_['max_iter'],
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
        
        # Save detailed grid search results
        detailed_csv = os.path.join(log_dir, f"detailed_{embedding_type}_{timestamp}.csv")
        results_df['embedding_type'] = embedding_type
        results_df['timestamp'] = timestamp
        results_df.to_csv(detailed_csv, index=False)
        logger.debug(f"[{embedding_type}] Дэлгэрэнгүй grid search үр дүн: {detailed_csv}")
        
        # Save best model
        model_file = os.path.join(models_dir, f"best_model_{embedding_type}_{timestamp}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(lr_grid.best_estimator_, f)
        logger.info(f"[{embedding_type}] Шилдэг модел: {model_file}")
        
        logger.info(f"[{embedding_type}] АМЖИЛТТАЙ ДУУСЛАА!")
        logger.info("="*80 + "\n")
        
        return lr_grid
        
    except Exception as e:
        logger.error(f"[{embedding_type}] АЛДАА ГАРЛАА: {str(e)}")
        logger.exception(f"[{embedding_type}] Дэлгэрэнгүй алдаа:")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Logistic Regression-ээр текстийн сэтгэл хөдлөл таних')
    parser.add_argument(
        '--embedding',
        type=str,
        default='bert_uncased',
        choices=['bert_uncased', 'bert_cased', 'roberta', 'albert', 'sbert', 'hatebert', 'word2vec'],
        help='Ашиглах эмбеддингийн төрөл (default: bert_uncased)'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default='config',
        help='Config файлуудын хавтас (default: config)'
    )
    
    args = parser.parse_args()
    
    # Load configurations
    paths_config = load_config(os.path.join(args.config_dir, 'paths.json'))
    lr_config = load_config(os.path.join(args.config_dir, 'logistic_regression_config.json'))
    
    # Load embeddings
    X, y = load_embeddings(args.embedding, paths_config)
    
    # Train model
    print(f"\nЭмбеддинг: {args.embedding.upper()}")
    model = LogisticRegress(X, y, args.embedding, lr_config, paths_config)

#     # BERT uncased (default)
# python src/models/logisticRegress.py --embedding bert_uncased

# # BERT cased
# python src/models/logisticRegress.py --embedding bert_cased

# # RoBERTa
# python src/models/logisticRegress.py --embedding roberta

# # ALBERT
# python src/models/logisticRegress.py --embedding albert

# # SBERT
# python src/models/logisticRegress.py --embedding sbert

# # HateBERT
# python src/models/logisticRegress.py --embedding hatebert

# # Word2Vec
# python src/models/logisticRegress.py --embedding word2vec