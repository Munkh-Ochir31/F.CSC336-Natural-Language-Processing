"""
BERT Embeddings Generator
Энэ скрипт нь янз бүрийн BERT загваруудыг ашиглан текстийн өгөгдлийг embedding болгоно.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import gc


class BERTEmbedder:
    """BERT загваруудаар embedding үүсгэх класс"""
    
    def __init__(self, device=None):
        """
        Args:
            device: Ашиглах төхөөрөмж ('cuda' эсвэл 'cpu')
        """
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Ашиглаж буй төхөөрөмж: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA хувилбар: {torch.version.cuda}")
        
    def bert_tokenizer(self, tokenizer, data, max_length):
        """
        Текстийг токен болгох
        
        Args:
            tokenizer: Токенизер
            data: Текстийн өгөгдөл
            max_length: Токений хамгийн их урт
            
        Returns:
            Токенуудын dictionary
        """
        token = {
            "input_ids": [],
            "attention_mask": []
        }
        
        for sent in data:
            encoded_sent = tokenizer.encode_plus(
                text=sent,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True
            )
            
            token["input_ids"].append(torch.tensor(encoded_sent["input_ids"]))
            token["attention_mask"].append(torch.tensor(encoded_sent["attention_mask"]))
        
        token["input_ids"] = torch.stack(token["input_ids"], dim=0)
        token["attention_mask"] = torch.stack(token["attention_mask"], dim=0)
        return token
    
    def word_embed(self, tokenizer, model, column, max_length, batch_size=8):
        """
        Текстийг embedding болгох
        
        Args:
            tokenizer: Токенизер
            model: BERT загвар
            column: Текстийн багана
            max_length: Токений хамгийн их урт
            batch_size: Batch хэмжээ
            
        Returns:
            Embedding tensor
        """
        dataloader = DataLoader(dataset=column, batch_size=batch_size, shuffle=False)
        otlst = []
        
        for batch_data in tqdm.tqdm(dataloader, desc="Embedding үүсгэж байна"):
            batch_token = self.bert_tokenizer(tokenizer, batch_data, max_length)
            
            for key in batch_token:
                batch_token[key] = batch_token[key].to(self.device)
            
            with torch.no_grad():
                outputs = model(**batch_token)
                emb = outputs.last_hidden_state
                emb = emb.half()  # float16 болгох (санах ой хэмнэх)
                emb = emb.cpu()   # CPU руу буцаах
            
            otlst.append(emb)
        
        sumrslt = torch.cat(otlst, dim=0)
        return sumrslt
    
    def generate_embeddings(self, data_path, output_dir, model_names=None, 
                          max_length=64, batch_size=8):
        """
        Өгөгдлийн файлаас embedding үүсгэх
        
        Args:
            data_path: Өгөгдлийн файлын зам
            output_dir: Хадгалах хавтасын зам
            model_names: Ашиглах загваруудын жагсаалт
            max_length: Токений хамгийн их урт
            batch_size: Batch хэмжээ
        """
        # Өгөгдөл уншиж авах
        print(f"\nӨгөгдөл уншиж байна: {data_path}")
        data = pd.read_csv(data_path)
        print(f"Өгөгдлийн хэмжээ: {len(data)}")
        
        X = data["review"]
        y = data["sentiment"]
        
        # y-г хадгалах
        Y = torch.tensor(y.values, dtype=torch.long)
        y_path = os.path.join(output_dir, "y.pt")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(Y, y_path)
        print(f"y хадгалагдлаа: {y_path}")
        
        # Загваруудын жагсаалт
        if model_names is None:
            model_names = {
                "bert_base_cased": "bert-base-cased",
                "bert_base_uncased": "google-bert/bert-base-uncased",
                "roberta": "FacebookAI/xlm-roberta-base",
                "sbert": "sentence-transformers/all-MiniLM-L6-v2",
                "hatebert": "GroNLP/hateBERT",
                "albert": "albert-base-v2"
            }
        
        # Загвар бүрээр embedding үүсгэх
        for key, model_name in model_names.items():
            try:
                print(f"\n{'='*60}")
                print(f"Загвар: {key} ({model_name})")
                print(f"{'='*60}")
                
                # Токенизер ба загвар ачаалах
                print("Токенизер ба загвар ачаалж байна...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                model = model.to(self.device)
                model.eval()
                
                # Embedding үүсгэх
                print(f"Embedding үүсгэж байна (max_length={max_length}, batch_size={batch_size})...")
                result = self.word_embed(tokenizer, model, X, max_length, batch_size)
                
                print(f"Embedding хэмжээ: {result.shape}")
                print(f"Өгөгдлийн төрөл: {result.dtype}")
                
                # Хадгалах
                output_path = os.path.join(output_dir, f"imdb_{key}_embed.pt")
                torch.save(result, output_path)
                print(f"✓ Хадгалагдлаа: {output_path}")
                
                # Санах ой чөлөөлөх
                del model
                del tokenizer
                del result
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"✗ Алдаа гарлаа ({key}): {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print("Бүх embedding үүсгэж дууслаа!")
        print(f"{'='*60}")


def main():
    """Үндсэн функц"""
    parser = argparse.ArgumentParser(
        description="BERT загваруудаар текст embedding үүсгэх"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="C:\\Users\\my tech\\Documents\\1. 3A\\NLP\\data\\preprocessed_data.csv",
        help="Өгөгдлийн файлын зам"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:\\Users\\my tech\\Documents\\1. 3A\\NLP\\data\\embedding\\embedded_x",
        help="Хадгалах хавтасын зам"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Ашиглах загваруудын нэрс (жнь: bert_base_cased roberta)"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Токений хамгийн их урт"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch хэмжээ"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Ашиглах төхөөрөмж (cuda эсвэл cpu)"
    )
    
    args = parser.parse_args()
    
    # Систем мэдээлэл хэвлэх
    print("="*60)
    print("BERT Embedding Generator")
    print("="*60)
    print(f"PyTorch хувилбар: {torch.__version__}")
    print(f"CUDA боломжтой: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA хувилбар: {torch.version.cuda}")
        print(f"GPU тоо: {torch.cuda.device_count()}")
    
    # Загваруудын жагсаалт бэлтгэх
    model_names = None
    if args.models:
        all_models = {
            "bert_base_cased": "bert-base-cased",
            "bert_base_uncased": "google-bert/bert-base-uncased",
            "roberta": "FacebookAI/xlm-roberta-base",
            "sbert": "sentence-transformers/all-MiniLM-L6-v2",
            "hatebert": "GroNLP/hateBERT",
            "albert": "albert-base-v2"
        }
        model_names = {k: v for k, v in all_models.items() if k in args.models}
    
    # Embedding үүсгэх
    embedder = BERTEmbedder(device=args.device)
    embedder.generate_embeddings(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_names=model_names,
        max_length=args.max_length,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
