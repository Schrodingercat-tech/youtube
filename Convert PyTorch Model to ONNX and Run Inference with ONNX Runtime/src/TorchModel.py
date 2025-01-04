import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# print("Current Working Directory:", os.getcwd())
# reading data 
data_path = "data/iris_dataset.csv"
data = pd.read_csv(data_path)

#shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# separate features and target
X = data.drop("target",axis=1)
y = data.target

#split the training and testing data 
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=69)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# network
model = nn.Sequential(
    nn.Linear(4, 32), 
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
    nn.Softmax(dim=1)  
)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        
        # Training loop
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()  # Clear gradients
            outputs = model(X_batch)  # Forward pass
            loss = criterion(outputs, y_batch)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()  # Accumulate loss
        
        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():  # No gradients needed for validation
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)  # Forward pass
                loss = criterion(outputs, y_batch)  # Compute loss
                val_loss += loss.item()  # Accumulate validation loss
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        if (epoch+1)%100==0:
        # Print loss and accuracy
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                f"Training Loss: {total_loss / len(train_loader):.4f}, "
                f"Validation Loss: {val_loss / len(val_loader):.4f}, "
                f"Validation Accuracy: {100 * correct / total:.2f}%")


train_model(model, train_loader,val_loader, criterion, optimizer, num_epochs=100)

# save the torch model
torch.save(model.state_dict(),"model/iris.pth")


