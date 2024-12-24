from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace, ByteLevel  # Import ByteLevel
from tokenizers.trainers import BpeTrainer

def train_bpe_tokenizer(corpus, vocab_size=1000):
    # Initialize a tokenizer with the BPE model
    tokenizer = Tokenizer(BPE())
    
    # Set the pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()  # You can switch to ByteLevel() if preferred
    
    # Initialize a trainer with the correct initial alphabet
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet()  # Use ByteLevel's alphabet
    )
    
    # Train the tokenizer on the provided corpus
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer

# Example Corpus
corpus = [
    "Hello! I'm learning about tokenization in NLP.",
    "Tokenization is a crucial step in preprocessing text data.",
    "Subword tokenization helps in handling unknown words."
]

# Train BPE Tokenizer with a larger vocab size for better coverage
bpe_tokenizer = train_bpe_tokenizer(corpus, vocab_size=50)  # You might want to increase vocab_size

# Encode a sample sentence
encoded = bpe_tokenizer.encode("Tokenization helps in handling unknown words effectively.")
print("BPE Encoded:", encoded.tokens)
