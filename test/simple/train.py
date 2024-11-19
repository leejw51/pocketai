import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import SineNet, degrees_to_radians

# Training parameters
EPISODES = 1000
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Initialize model and optimizer
model = SineNet()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Training loop
losses = []
for episode in range(EPISODES):
    # Generate random angles in degrees
    degrees = np.random.uniform(0, 360, BATCH_SIZE)
    
    # Convert to tensors and prepare input
    x = torch.FloatTensor(degrees).view(-1, 1)
    
    # Calculate true sine values (convert degrees to radians first)
    y_true = torch.sin(torch.FloatTensor([degrees_to_radians(d) for d in degrees])).view(-1, 1)
    
    # Forward pass
    y_pred = model(x / 360.0)  # Normalize input
    
    # Calculate loss
    loss = criterion(y_pred, y_true)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (episode + 1) % 100 == 0:
        print(f'Episode [{episode+1}/{EPISODES}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'sine_model.pth')

# Plot training loss
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.show() 