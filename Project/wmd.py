from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.similarities import WmdSimilarity
import numpy as np

# Example data
questions = ["What is the capital of France?", "Who painted the Mona Lisa?"]
answers = ["The capital of France is Paris.", "Leonardo da Vinci painted the Mona Lisa."]

# Load pre-trained word vectors (such as Word2Vec or GloVe)
word_vectors = KeyedVectors.load_word2vec_format('path/to/word2vec.bin', binary=True)

# Preprocess the text
tokenized_questions = [question.lower().split() for question in questions]
tokenized_answers = [answer.lower().split() for answer in answers]

# Calculate WMD similarity
wmd_similarities = []
for answer in tokenized_answers:
    for question in tokenized_questions:
        wmd_similarity = word_vectors.wmdistance(question, answer)
        wmd_similarities.append(wmd_similarity)

# You can set a threshold for what constitutes an accurate answer based on WMD distance
threshold = 1.0
predictions = ["Accurate" if similarity <= threshold else "Inaccurate" for similarity in wmd_similarities]

# Print predictions
for i, prediction in enumerate(predictions):
    print(f"Answer: '{answers[i]}', Prediction: {prediction}, WMD Similarity: {wmd_similarities[i]}")
