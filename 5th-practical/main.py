# Sample Corpus
corpus = [
    "I love natural language processing in all chatbot for personal use",
    "Word embeddings are useful for NLP tasks",
    "Deep learning models perform well on language data",
    "Natural language understanding is a key aspect of AI",
    "Embeddings capture semantic relationships between words",
    "Machine learning enables computers to learn from data",
    "AI and NLP are rapidly evolving fields",
    "Understanding context is crucial for language models",
    "Language models generate human-like text",
    "Semantic analysis helps in extracting meaningful information"
    "all men are equals",
    "all women also equals"
]

import numpy as np
import re

def preprocess_corpus(corpus):
    """
    Preprocesses the corpus by lowercasing, removing non-alphabetic characters, and tokenizing.
    
    :param corpus: List of sentences (strings)
    :return: List of tokenized sentences
    """
    processed_corpus = []
    for sentence in corpus:
        # Lowercase
        sentence = sentence.lower()
        # Remove non-alphabetic characters
        sentence = re.sub(r'[^a-z\s]', '', sentence)
        # Tokenize
        tokens = sentence.split()
        processed_corpus.append(tokens)
    return processed_corpus

processed_corpus = preprocess_corpus(corpus)
# print("Processed Corpus:")
# for sentence in processed_corpus:
#     print(sentence)
    
    
from collections import defaultdict

def build_vocabulary(processed_corpus):
    """
    Builds a vocabulary mapping each unique word to a unique index.
    
    :param processed_corpus: List of tokenized sentences
    :return: word_to_index (dict), index_to_word (dict)
    """
    word_freq = defaultdict(int)
    for sentence in processed_corpus:
        for word in sentence:
            word_freq[word] += 1
    
    # Sort words by frequency
    sorted_words = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    
    word_to_index = {word: idx for idx, (word, _) in enumerate(sorted_words)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    
    return word_to_index, index_to_word, word_freq

word_to_index, index_to_word, word_freq = build_vocabulary(processed_corpus)

# print("\nVocabulary Size:", len(word_to_index))
# print("Word to Index Mapping:")
# for word, idx in word_to_index.items():
#     print(f"{word}: {idx}")

def generate_training_pairs(processed_corpus, word_to_index, window_size=2):
    """
    Generates training pairs for the CBOW model.
    
    :param processed_corpus: List of tokenized sentences
    :param word_to_index: Dictionary mapping words to indices
    :param window_size: Number of context words on each side
    :return: List of (context_indices, target_index) tuples
    """
    training_pairs = []
    for sentence in processed_corpus:
        sentence_length = len(sentence)
        for idx, target_word in enumerate(sentence):
            context = []
            # Define context window
            for i in range(idx - window_size, idx + window_size + 1):
                if i != idx and 0 <= i < sentence_length:
                    context.append(word_to_index[sentence[i]])
            if context:
                training_pairs.append((context, word_to_index[target_word]))
    return training_pairs

training_pairs = generate_training_pairs(processed_corpus, word_to_index, window_size=2)
# print("\nSample Training Pairs:")
# for pair in training_pairs[:8]:
#     context_words = [index_to_word[idx] for idx in pair[0]]
#     target_word = index_to_word[pair[1]]
#     print(f"Context: {context_words} -> Target: {target_word}")

class CBOWModel:
    def __init__(self, vocab_size, embedding_dim):
        """
        Initializes the CBOW model.
        
        :param vocab_size: Size of the vocabulary
        :param embedding_dim: Dimension of the embedding vectors
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize weights
        self.W1 = np.random.randn(vocab_size, embedding_dim)
        print(self.W1)
        self.W2 = np.random.randn(embedding_dim, vocab_size)
        
    def forward(self, context_indices):
        """
        Forward pass: Computes the predicted output probabilities.
        
        :param context_indices: List of context word indices
        :return: Predicted probabilities (softmax output)
        """
        # Average the context word vectors
        h = np.mean(self.W1[context_indices], axis=0)  # Shape: (embedding_dim,)
        # Compute scores
        u = np.dot(h, self.W2)  # Shape: (vocab_size,)
        # Apply softmax
        y_pred = self.softmax(u)
        return y_pred, h
    
    def softmax(self, x):
        """
        Computes the softmax of a vector.
        
        :param x: Input vector
        :return: Softmax output vector
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def backward(self, y_pred, target_index, h, context_indices, learning_rate=0.01):
        """
        Backward pass: Updates the weights based on the error.
        
        :param y_pred: Predicted probabilities
        :param target_index: Index of the target word
        :param h: Hidden layer output (average of context word vectors)
        :param context_indices: List of context word indices
        :param learning_rate: Learning rate for weight updates
        """
        # Initialize gradient vectors
        e = np.zeros(self.vocab_size)
        e[target_index] = 1
        # Compute output layer error
        delta = y_pred - e  # Shape: (vocab_size,)
        # Compute gradients for W2
        dW2 = np.outer(h, delta)  # Shape: (embedding_dim, vocab_size)
        # Compute gradients for W1
        dh = np.dot(self.W2, delta) / len(context_indices)  # Shape: (embedding_dim,)
        dW1 = np.zeros_like(self.W1)
        for idx in context_indices:
            dW1[idx] += dh  # Shape: (vocab_size, embedding_dim)
        
        # Update weights
        self.W1 -= learning_rate * dW1
        self.W2 -= learning_rate * dW2
    
    def train(self, training_pairs, epochs=1000, learning_rate=0.01):
        """
        Trains the CBOW model using the training pairs.
        
        :param training_pairs: List of (context_indices, target_index) tuples
        :param epochs: Number of training iterations
        :param learning_rate: Learning rate for weight updates
        """
        for epoch in range(1, epochs + 1):
            loss = 0
            for context_indices, target_index in training_pairs:
                y_pred, h = self.forward(context_indices)
                # Compute loss (cross-entropy)
                loss -= np.log(y_pred[target_index] + 1e-7)
                # Backward pass
                self.backward(y_pred, target_index, h, context_indices, learning_rate)
            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
    
    def get_embeddings(self):
        """
        Retrieves the word embeddings.
        
        :return: Embedding matrix (W1)
        """
        return self.W1
    
    
# Parameters
vocab_size = len(word_to_index)
embedding_dim = 10  # You can experiment with different dimensions
epochs = 1000
learning_rate = 0.01

print("vocab_size :", vocab_size)

# Initialize the CBOW model
cbow = CBOWModel(vocab_size, embedding_dim)

# Train the model
# cbow.train(training_pairs, epochs=epochs, learning_rate=learning_rate)

# Retrieve the embeddings
# embeddings = cbow.get_embeddings()
# print("\nWord Embeddings:")
# for word, idx in word_to_index.items():
#     print(f"{word}: {embeddings[idx]}")