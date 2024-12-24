import spacy

nlp = spacy.load("en_core_web_sm")
text = "I wouldn't face any 12: problem."
tokens = [token.text for token in nlp(text)]
print(tokens)
