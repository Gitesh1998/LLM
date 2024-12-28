from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

a = ["a b c d e f g h i", "d f e c v h i" ]

tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(a), specials=["<unk>"])

# print(vocab)
# print(len(vocab))
# print(vocab.lookup_tokens([0]))

# for tokens in yield_tokens(a):
#     print(tokens)

for i in range(len(vocab)):
    print(vocab.lookup_tokens([i]))