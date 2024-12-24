import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import re
import matplotlib.pyplot as plt

# Initialize SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize Stemmer and Lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define stop words
stop_words = set(stopwords.words('english'))

def nltk_word_tokenize(text):
    return word_tokenize(text)

# Example
sample_text = "Hello! I'm learning about tokenization in NLP. How to perform operations"
tokens = nltk_word_tokenize(sample_text)
print("NLTK Word Tokens:", tokens)

def spacy_word_tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]

# Example
tokens = spacy_word_tokenize(sample_text)
print("SpaCy Word Tokens:", tokens)

def nltk_sentence_tokenize(text):
    return sent_tokenize(text)

# Example
sentences = nltk_sentence_tokenize(sample_text)
print("NLTK Sentences:", sentences)

def spacy_sentence_tokenize(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# Example
sentences = spacy_sentence_tokenize(sample_text)
print("SpaCy Sentences:", sentences)

import contractions

def expand_contractions_text(text):
    return contractions.fix(text)

# Example
contracted_text = "Im excited to learn about NLP and its applications!"
expanded_text = expand_contractions_text(contracted_text)
print("Expanded Text:", expanded_text)