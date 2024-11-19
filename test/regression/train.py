import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import RegressionModel


# Custom Dataset
class RegressionDataset(Dataset):
    def __init__(self, num_samples=1000):
        # Generate random data: y = 2x + 1 + noise
        self.x = torch.randn(num_samples, 1)
        self.y = 2 * self.x + 1 + 0.1 * torch.randn(num_samples, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def show_dataset_sample(dataset, num_samples=10):
    print("\nDataset Sample:")
    print("X\t\tY (2x + 1 + noise)")
    print("-" * 30)
    for i in range(num_samples):
        x, y = dataset[i]
        print(f"{x.item():0.3f}\t\t{y.item():0.3f}")


def train():
    # Hyperparameters
    learning_rate = 0.01
    num_epochs = 100
    batch_size = 32

    # Create dataset and dataloader
    dataset = RegressionDataset()

    # Show sample of the dataset before training
    show_dataset_sample(dataset)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = RegressionModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}"
            )

    # Save the trained model
    torch.save(model.state_dict(), "regression_model.pth")


if __name__ == "__main__":
    train()
