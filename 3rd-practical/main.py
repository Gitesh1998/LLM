import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
import re


# Initialize SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize Stemmer and Lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define stop words
stop_words = set(stopwords.words('english'))
stop_words.discard('i')


def remove_special_characters(text):
    pattern = r'[^a-zA-Z\s]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def tokenize(text):
    return word_tokenize(text)

def remove_stop_words(tokens):
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def stem_tokens(tokens):
    stemmed = [stemmer.stem(word) for word in tokens]
    return stemmed

def lemmatize_tokens(tokens):
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized

import contractions

def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text

sample_text = "I wouldn't facing any 12 : problem"
# contract_string = expand_contractions(sample_text)
# clean_word = remove_special_characters(contract_string)
# tokens = tokenize(clean_word)
# rm_stop = remove_stop_words(tokens)
# stem = stem_tokens(rm_stop)
# lemma_word = lemmatize_tokens(stem)

# print(lemma_word)

def preprocess_text(text, method='lemmatization'):
    text = expand_contractions(text)
    text = remove_special_characters(text)
    text = text.lower()
    tokens = tokenize(text)
    tokens = remove_stop_words(tokens)
    if method == 'stemming':
        tokens = stem_tokens(tokens)
    elif method == 'lemmatization':
        tokens = lemmatize_tokens(tokens)
    else:
        raise ValueError("Method must be 'stemming' or 'lemmatization'")
    
    return tokens

print(preprocess_text(sample_text))