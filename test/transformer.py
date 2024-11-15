import torch
import torch.nn as nn
import torch.optim as optim
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # pe shape: [seq_len, d_model]
        return x + self.pe[:x.size(1)]

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=2, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Simple transformer with reduced parameters
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                 dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # Input shape: [batch_size, seq_len]
        print(f"Input shape: {x.shape}")
        
        # Embedding shape: [batch_size, seq_len, d_model]
        x = self.embedding(x)
        print(f"After embedding shape: {x.shape}")
        
        # Add positional encoding
        x = self.pos_encoder(x)
        print(f"After positional encoding shape: {x.shape}")
        
        # Transformer shape: [batch_size, seq_len, d_model]
        x = self.transformer(x)
        print(f"After transformer shape: {x.shape}")
        
        # Output shape: [batch_size, seq_len, vocab_size]
        x = self.fc_out(x)
        print(f"Output shape: {x.shape}")
        
        return x

def create_vocabulary(sentences):
    """Create a simple vocabulary from the sentences"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def tokenize(sentence, vocab, seq_length):
    """Convert a sentence to token ids with padding"""
    words = sentence.split()
    tokens = [vocab.get(word, vocab['<UNK>']) for word in words]
    # Pad or truncate to seq_length
    if len(tokens) < seq_length:
        tokens = tokens + [vocab['<PAD>']] * (seq_length - len(tokens))
    else:
        tokens = tokens[:seq_length]
    return tokens

def train_example():
    # Example sentences
    sentences = [
        "hello how are you",
        "i am doing well",
        "nice to meet you",
        "what a beautiful day",
        "the weather is nice",
        "i love programming",
        "python is awesome",
        "transformers are cool",
        "deep learning is fun",
        "have a great day"
    ]
    
    # Parameters
    seq_length = 8  # Maximum sequence length
    batch_size = 5  # Process 5 sentences at once
    
    # Create vocabulary from sentences
    vocab = create_vocabulary(sentences)
    vocab_size = len(vocab)
    print("\nVocabulary:")
    print(vocab)
    
    # Convert sentences to token ids
    tokenized_data = [tokenize(sent, vocab, seq_length) for sent in sentences]
    input_data = torch.tensor(tokenized_data)
    
    # For this example, target will be the same as input (like autoencoder)
    target_data = input_data.clone()
    
    # Create model
    model = SimpleTransformer(vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining Example:")
    print("Input sentences (first batch):")
    for i in range(batch_size):
        sent = sentences[i]
        tokens = tokenized_data[i]
        print(f"Sentence {i+1}: '{sent}'")
        print(f"Tokenized: {tokens}")
    
    # Training loop
    model.train()
    for epoch in range(2):
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch_input = input_data[i:i+batch_size]
            batch_target = target_data[i:i+batch_size]
            
            optimizer.zero_grad()
            
            print(f"\nBatch {i//batch_size + 1}:")
            output = model(batch_input)
            
            loss = criterion(output.view(-1, vocab_size), batch_target.view(-1))
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}, Batch {i//batch_size + 1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_example()
