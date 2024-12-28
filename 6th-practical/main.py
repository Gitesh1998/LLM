import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
# from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
# import os
from tqdm import tqdm

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define tokenizer
tokenizer = get_tokenizer('basic_english')

# Load training and test data
train_iter, test_iter = IMDB(split=('train', 'test'))

# Function to yield tokens
def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

# Build vocabulary
from torchtext.vocab import build_vocab_from_iterator

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# print(vocab)

# Reset train and test iterators
train_iter, test_iter = IMDB(split='train'), IMDB(split='test')

# for i, example in enumerate(train_iter):
#     label, text = example
#     print(f"Example {i + 1}")
#     print(f"Label: {label}")
#     print(f"Text: {text[:200]}...")  # Print the first 200 characters of the text
#     print("-" * 80)
#     if i == 4:  # Stop after 5 examples
#         break

# Define text processing pipeline
def process_text(text):
    return vocab(tokenizer(text))

# Example
# label, text = next(iter(train_iter))
# print(f"Label: {label}")
# print(f"Processed Text: {process_text(text[:50])} ...")

class IMDBDataset(Dataset):
    def __init__(self, data_iter, vocab, tokenizer):
        self.labels = []
        self.texts = []
        self.vocab = vocab
        self.tokenizer = tokenizer
        for label, text in data_iter:
            self.labels.append(1 if label == 'pos' else 0)
            self.texts.append(torch.tensor(self.vocab(self.tokenizer(text)), dtype=torch.long))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Create datasets
train_dataset = IMDBDataset(train_iter, vocab, tokenizer)
test_dataset = IMDBDataset(test_iter, vocab, tokenizer)

# Define collate function for padding
def collate_batch(batch):
    text_list, label_list = zip(*batch)
    text_lengths = [len(text) for text in text_list]
    padded_texts = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)
    labels = torch.tensor(label_list, dtype=torch.float32)
    return padded_texts.to(device), labels.to(device), torch.tensor(text_lengths, dtype=torch.long).to(device)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, lengths):
        # x: [batch_size, seq_length]
        embedded = self.embedding(x)  # [batch_size, seq_length, embed_dim]
        
        # Pack the sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Pass through LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack the sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output: [batch_size, seq_length, hidden_dim * num_directions]
        
        # Concatenate the final forward and backward hidden states
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # [batch_size, hidden_dim * 2]
        else:
            hidden = hidden[-1,:,:]  # [batch_size, hidden_dim]
        
        # Pass through the fully connected layer
        out = self.fc(hidden)  # [batch_size, 1]
        out = self.sigmoid(out)  # [batch_size, 1]
        return out.squeeze()


# Model parameters
vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 256
num_layers = 2
bidirectional = True
dropout = 0.5
num_epochs = 5
learning_rate = 0.001

# Initialize the model, loss function, and optimizer
model = LSTMModel(vocab_size, embed_dim, hidden_dim, num_layers, bidirectional, dropout).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to calculate metrics
def compute_metrics(preds, labels):
    preds = (preds >= 0.5).float()
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for texts, labels, lengths in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * texts.size(0)
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += texts.size(0)
    
    avg_loss = epoch_loss / total
    accuracy = correct / total * 100
    print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")


model.eval()
all_preds = []
all_labels = []
test_loss = 0

with torch.no_grad():
    for texts, labels, lengths in tqdm(test_loader, desc="Evaluating"):
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * texts.size(0)
        preds = (outputs >= 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Concatenate all predictions and labels
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Compute metrics
metrics = compute_metrics(all_preds, all_labels)
avg_test_loss = test_loss / len(test_loader.dataset)
print(f"\nTest Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Test Precision: {metrics['precision']:.4f}")
print(f"Test Recall: {metrics['recall']:.4f}")
print(f"Test F1 Score: {metrics['f1']:.4f}")

print("Successful")