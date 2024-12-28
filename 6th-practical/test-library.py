import torch
import torchtext

print(f"PyTorch version: {torch.__version__}")
print(f"TorchText version: {torchtext.__version__}")

import torch
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")


# Test importing IMDB dataset
from torchtext.datasets import IMDB

train_iter, test_iter = IMDB(split=('train', 'test'))
print("IMDB dataset loaded successfully.")
