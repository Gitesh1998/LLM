import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # For lemmatization

# Download SpaCy English model
import spacy
spacy.cli.download("en_core_web_sm")