# BERT Models Sentiment Analysis - Usage Guide

Энэ guide нь `src/models/` дотор байгаа BERT модулиудыг хэрхэн ашиглах талаар тайлбарласан.

## 📁 Файлын бүтэц

```
src/
├── models/
│   ├── bert_models.py    # BERT загваруудын классууд
│   ├── predict.py        # Prediction скрипт
│   └── train.py          # Training скрипт
```

## 🚀 Хэрэглээний жишээ

### 1. Python код дотор ашиглах

#### Энгийн prediction:

```python
from src.models.bert_models import BERTSentimentModel

# BERT загвар ачаалах
model = BERTSentimentModel('bert-base-uncased')

# Текст таах
text = "This movie is fantastic!"
result = model.predict(text)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

**Гаралт:**
```
Sentiment: positive
Confidence: 98.50%
Probabilities: {'negative': 0.015, 'positive': 0.985}
```

#### Олон текст prediction:

```python
texts = [
    "Great movie!",
    "Terrible waste of time",
    "It was okay"
]

results = model.predict_batch(texts)

for text, result in zip(texts, results):
    print(f"{text} -> {result['sentiment']} ({result['confidence']:.2%})")
```

#### Бүх BERT загваруудыг харьцуулах:

```python
from src.models.bert_models import BERTSentimentModel, compare_models

# Загварууд үүсгэх
models = {
    'BERT': BERTSentimentModel('bert-base-uncased'),
    'BERT Cased': BERTSentimentModel('bert-base-cased'),
    'RoBERTa': BERTSentimentModel('roberta-base'),
    'ALBERT': BERTSentimentModel('albert-base-v2'),
    'HateBERT': BERTSentimentModel('GroNLP/hateBERT')
}

# Харьцуулах
text = "This movie is amazing!"
results = compare_models(text, models)

for name, result in results.items():
    print(f"{name}: {result['sentiment']} ({result['confidence']:.2%})")
```

#### SBERT ашиглах:

```python
from src.models.bert_models import SBERTSentimentModel

# SBERT загвар + classifier
model = SBERTSentimentModel('all-MiniLM-L6-v2')
model.load_classifier('./notebooks/sbert_classifier.pkl')

result = model.predict("This is a great film!")
print(result)
```

### 2. Command line ашиглах

#### Нэг текст prediction:

```bash
# BERT-ээр prediction
python src/models/predict.py --text "This movie is amazing!" --model bert-base-uncased

# RoBERTa-аар prediction
python src/models/predict.py --text "Terrible movie" --model roberta-base

# HateBERT-ээр prediction
python src/models/predict.py --text "This is garbage" --model GroNLP/hateBERT
```

#### Бүх загваруудыг харьцуулах:

```bash
python src/models/predict.py --text "This movie is fantastic!" --compare
```

**Гаралт:**
```
================================================================================
Comparing all BERT models...
================================================================================
Text: This movie is fantastic!

Loading BERT Base Uncased...
  ✓ positive (confidence: 99.12%)
Loading BERT Base Cased...
  ✓ positive (confidence: 98.87%)
Loading RoBERTa...
  ✓ positive (confidence: 99.45%)
Loading ALBERT...
  ✓ positive (confidence: 97.23%)
Loading HateBERT...
  ✓ positive (confidence: 98.91%)

================================================================================
Summary:
================================================================================
🟢 BERT Base Uncased   : positive   (99.12%)
🟢 BERT Base Cased     : positive   (98.87%)
🟢 RoBERTa             : positive   (99.45%)
🟢 ALBERT              : positive   (97.23%)
🟢 HateBERT            : positive   (98.91%)
```

#### Олон текст file-аас унших:

```bash
# texts.txt файл үүсгэх
echo "This movie is great!" > texts.txt
echo "Terrible waste of time" >> texts.txt
echo "It was okay" >> texts.txt

# Prediction хийх
python src/models/predict.py --file texts.txt --model bert-base-uncased --output results.txt
```

#### Interactive горим:

```bash
python src/models/predict.py --interactive
```

**Жишээ:**
```
================================================================================
BERT Sentiment Analysis - Interactive Mode
================================================================================

Available models:
  1. BERT Base Uncased
  2. BERT Base Cased
  3. RoBERTa
  4. ALBERT
  5. HateBERT
  6. Compare All

Type 'quit' to exit

Select model (1-6): 1
Enter text: This movie is amazing!

🟢 Prediction: POSITIVE
   Confidence: 99.12%
   Probabilities:
     - Negative: 0.88%
     - Positive: 99.12%
```

### 3. Сургасан модел ашиглах

Colab дээр сургаад татаж авсан модел:

```python
from src.models.bert_models import BERTSentimentModel

# Сургасан модел ачаалах
model = BERTSentimentModel('./notebooks/bert_sentiment')

# Prediction хийх
result = model.predict("This is a great movie!")
print(result)
```

Command line:

```bash
python src/models/predict.py \
  --text "Great movie!" \
  --model-path ./notebooks/bert_sentiment
```

### 4. Шинэ модел сургах

```bash
# BERT сургах
python src/models/train.py \
  --data data/cleaned_label.csv \
  --model bert-base-uncased \
  --output ./models/my_bert \
  --epochs 3 \
  --batch-size 16

# RoBERTa сургах
python src/models/train.py \
  --data data/cleaned_label.csv \
  --model roberta-base \
  --output ./models/my_roberta \
  --epochs 3

# ALBERT сургах
python src/models/train.py \
  --data data/cleaned_label.csv \
  --model albert-base-v2 \
  --output ./models/my_albert \
  --epochs 3
```

## 📊 Бүх 6 загварын хамрах хүрээ

| Загвар | Model Name | Parameters | Онцлог |
|--------|-----------|-----------|--------|
| BERT Base Uncased | `bert-base-uncased` | 110M | Стандарт BERT, жижиг үсгээр |
| BERT Base Cased | `bert-base-cased` | 110M | Том жижиг үсэг ялгадаг |
| RoBERTa | `roberta-base` | 125M | Сайжруулсан BERT |
| SBERT | `all-MiniLM-L6-v2` | 22M | Sentence embeddings |
| ALBERT | `albert-base-v2` | 12M | Жижиг, хурдан |
| HateBERT | `GroNLP/hateBERT` | 110M | Toxic content-д сайн |

## 🔧 Давуу талууд

### BERT Base Uncased
- ✅ Стандарт, өргөн хэрэглэгддэг
- ✅ Сайн ерөнхий үр дүн

### BERT Base Cased
- ✅ Том жижиг үсэг ялгадаг
- ✅ Нэр томъёо сайн таньдаг

### RoBERTa
- ✅ BERT-ээс сайн
- ✅ Илүү сайн pretraining

### SBERT
- ✅ Маш хурдан (500x faster than BERT)
- ✅ Sentence similarity
- ✅ Semantic search

### ALBERT
- ✅ Хамгийн жижиг (12M parameters)
- ✅ Хамгийн хурдан
- ✅ Parameter sharing

### HateBERT
- ✅ Сөрөг сэтгэгдэл илүү сайн
- ✅ Toxic/offensive language
- ✅ Hate speech detection

## 💡 Зөвлөмж

1. **Энгийн sentiment analysis**: BERT Base Uncased
2. **Хурдан inference**: ALBERT эсвэл SBERT
3. **Сөрөг сэтгэгдэл илүү сайн**: HateBERT
4. **Ерөнхий сайн үр дүн**: RoBERTa
5. **Semantic search/similarity**: SBERT

## 🎯 Өргөн хэрэглээний жишээ

### Web API үүсгэх:

```python
from flask import Flask, request, jsonify
from src.models.bert_models import BERTSentimentModel

app = Flask(__name__)
model = BERTSentimentModel('roberta-base')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')
    result = model.predict(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

### Batch processing:

```python
import pandas as pd
from src.models.bert_models import BERTSentimentModel

# CSV унших
df = pd.read_csv('reviews.csv')

# Модел ачаалах
model = BERTSentimentModel('roberta-base')

# Batch prediction
results = model.predict_batch(df['review_text'].tolist(), batch_size=32)

# Үр дүн нэмэх
df['sentiment'] = [r['sentiment'] for r in results]
df['confidence'] = [r['confidence'] for r in results]

# Хадгалах
df.to_csv('reviews_with_sentiment.csv', index=False)
```

## 🐛 Troubleshooting

**CUDA out of memory:**
```python
# Batch size-г багасгах
model.predict_batch(texts, batch_size=8)
```

**Модел олдохгүй:**
```python
# Эхлээд татаж авах
from transformers import AutoModel
AutoModel.from_pretrained('bert-base-uncased')
```

**Import алдаа:**
```python
import sys
sys.path.append('/path/to/biydaalt1')
from src.models.bert_models import BERTSentimentModel
```

Амжилт хүсье! 🚀
