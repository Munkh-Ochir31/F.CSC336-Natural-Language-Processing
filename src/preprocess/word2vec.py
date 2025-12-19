import nltk
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import pandas as pd

# Ensure necessary NLTK data is downloaded for tokenization
nltk.download('punkt')

# 3. Train the Word2Vec Model
# Initialize and train the model using the processed sentences
# sg=1 specifies the Skip-Gram model (sg=0 for CBOW)
model = Word2Vec(
    sentences=processed_sentences,
    vector_size=100,  # Dimensionality of the word vectors
    window=5,         # Context window size
    min_count=1,      # Ignores all words with total frequency lower than this
    sg=1              # Use Skip-Gram
)

# 4. Access and Use the Word Vectors
# Get the vector for a specific word
word_vector = model.wv['learning']
print(f"Vector for 'learning' (first 5 dimensions): {word_vector[:5]}\n")

# Find the most similar words to a given word
similar_words = model.wv.most_similar('learning', topn=3)
print(f"Words similar to 'learning': {similar_words}\n")

# Calculate the similarity between two words
similarity = model.wv.similarity('machine', 'AI')
print(f"Similarity between 'machine' and 'AI': {similarity}")

# 5. Save the trained model for later use
model.save("word2vec.model")
print("\nModel saved to word2vec.model")

def wod2vec(data_path, vector_size, window, min_count,sg,):

    data = pd.read_csv(data_path)
    model = Word2Vec(
        data=data,
        vector_size=vector_size,
        window=window,   
        min_count=min_count,
        sg=sg
    )
    word_vector = model.wv['learning']
    print(f"Vector for 'learning' (first 5 dimensions): {word_vector[:5]}\n")