import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import GPT2, GPT2Config
from collections import Counter
import numpy as np

# Sample dataset
texts = [
    "hello world",
    "how are you",
    "nice to meet you",
    "have a great day",
    "what is your name",
    "the weather is nice",
    "i love programming",
    "python is awesome",
    "deep learning is fun",
    "goodbye see you later",
]


# Tokenization (character-level for simplicity)
class CharacterTokenizer:
    def __init__(self, texts):
        self.chars = sorted(list(set("".join(texts))))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        return "".join([self.idx_to_char[idx] for idx in indices])


# Dataset
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) < max_length:
                tokens = tokens + [0] * (max_length - len(tokens))
            self.data.append(tokens[:max_length])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


# Initialize tokenizer and dataset
tokenizer = CharacterTokenizer(texts)
dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Modify the device selection to include MPS
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)
print(f"Using device: {device}")

# Initialize model with GPT-2 config
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    block_size=32,  # max sequence length
    n_layer=4,  # number of transformer blocks
    n_head=4,  # number of attention heads
    n_embd=128,  # embedding dimension
    dropout=0.1,
    bias=True,
)

model = GPT2(config)
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train(epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


# Inference function
def generate_text(prompt, max_length=32, temperature=0.8):
    model.eval()
    with torch.no_grad():
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor(tokens).unsqueeze(0).to(device)

        for _ in range(max_length - len(prompt)):
            logits = model(tokens)
            next_token_logits = logits[0, -1, :] / temperature
            # Apply softmax to convert logits to probabilities
            probs = nn.functional.softmax(next_token_logits, dim=-1)
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens = torch.cat([tokens, torch.tensor([[next_token]]).to(device)], dim=1)

            if next_token == 0:
                break

        return tokenizer.decode(tokens[0].cpu().numpy())


# Save the model after training
print("Training started...")
train()
torch.save(model.state_dict(), "gpt2_model.pt")

# Generate some text
print("\nGenerating text...")
prompts = ["hello", "how", "the"]
for prompt in prompts:
    generated = generate_text(prompt)
    print(f"Prompt: '{prompt}' → Generated: '{generated}'")
