import os
import pandas as pd
import re
from pathlib import Path

def clean_text(text):
    """Clean and normalize text"""
    text = re.sub(r'<br\s*/?>','', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def process_aclimdb_dataset(base_path):
    """Process the IMDB dataset from aclImdb directory"""
    
    data = []
    base_path = Path(base_path)
    for split in ['train', 'test']:
        split_path = base_path / split
        for sentiment in ['pos', 'neg']:
            sentiment_path = split_path / sentiment
            
            if not sentiment_path.exists():
                print(f"Warning: {sentiment_path} does not exist")
                continue
            
            print(f"Processing {split}/{sentiment}...")
            
            # Get all text files
            files = list(sentiment_path.glob('*.txt'))
            
            for idx, file_path in enumerate(files):
                try:
                    filename = file_path.stem
                    parts = filename.split('_')
                    review_id = parts[0]
                    rating = int(parts[1]) if len(parts) > 1 else None
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        review_text = f.read()
                    
                    cleaned_text = clean_text(review_text)
                    
                    data.append({
                        'review_id': review_id,
                        'split': split,
                        'sentiment': sentiment,
                        'rating': rating,
                        'review_text': cleaned_text
                    })
                    
                    if (idx + 1) % 1000 == 0:
                        print(f"  Processed {idx + 1} files...")
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue
    
    return data

def main():
    base_path = '/home/tr1bo/Documents/4. 3A/Эх хэлний боловсруулалт/biydaalt1/data/aclImdb'
    
    print("Starting data preprocessing...")
    print("=" * 60)
    
    data = process_aclimdb_dataset(base_path)
    
    print("\n" + "=" * 60)
    print(f"Total reviews processed: {len(data)}")
    
    df = pd.DataFrame(data)
    df['sentiment_label'] = df['sentiment'].map({'neg': 0, 'pos': 1})
    df = df[['review_id', 'split', 'sentiment', 'sentiment_label', 'rating', 'review_text']]
    df = df.sort_values(['split', 'sentiment', 'review_id']).reset_index(drop=True)
    
    print("\nDataset Statistics:")
    print("-" * 60)
    print(f"Total reviews: {len(df)}")
    print(f"\nBy split:")
    print(df['split'].value_counts())
    print(f"\nBy sentiment:")
    print(df['sentiment'].value_counts())
    print(f"\nBy split and sentiment:")
    print(df.groupby(['split', 'sentiment']).size())

    output_path = '/home/tr1bo/Documents/4. 3A/Эх хэлний боловсруулалт/biydaalt1/data/cleaned.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n{'=' * 60}")
    print(f"Data saved to: {output_path}")
    print(f"CSV file size: {len(df)} rows x {len(df.columns)} columns")
    print("\nFirst few rows:")
    print(df.head())
    print("\nPreprocessing complete!")

def create_text_label_subset():
    """Create a subset CSV with only review_text and sentiment_label columns"""
    input_file = '/home/tr1bo/Documents/4. 3A/Эх хэлний боловсруулалт/biydaalt1/data/cleaned.csv'
    output_file = '/home/tr1bo/Documents/4. 3A/Эх хэлний боловсруулалт/biydaalt1/data/cleaned_label.csv'
    
    print("\nReading CSV file...")
    df = pd.read_csv(input_file)
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    df_subset = df[['review_text', 'sentiment_label']]
    
    print(f"\nNew shape: {df_subset.shape}")
    print(f"New columns: {list(df_subset.columns)}")
    
    print("\nFirst few rows:")
    print(df_subset.head())
    
    print("\nLabel distribution:")
    print(df_subset['sentiment_label'].value_counts())
    
    df_subset.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nNew CSV saved to: {output_file}")
    print(f"Total rows: {len(df_subset)}")

if __name__ == '__main__':
    create_text_label_subset()