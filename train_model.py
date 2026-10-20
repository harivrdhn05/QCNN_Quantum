import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time  # <--- NEW: Added for time tracking
from torch.utils.data import TensorDataset, DataLoader

# --- CONFIGURATION ---
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.001
# ---------------------

# 1. Define the Classifier Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input shape: 4 channels (from quantum), 32x32 size
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1) 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) # Reduces to 16x16
        self.fc1 = nn.Linear(16 * 16 * 16, 2) # 2 outputs: Drought (0) or Healthy (1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    print(" Loading Quantum Features...")
    
    # Load the Pre-Scanned Data
    try:
        X_train = np.load("X_train_quantum.npy").astype(np.float32)
        X_test = np.load("X_test_quantum.npy").astype(np.float32)
        y_train = np.load("y_train.npy").astype(np.int64)
        y_test = np.load("y_test.npy").astype(np.int64)
    except:
        print(" Error: Run 'quantum_scan.py' first!")
        exit()

    # Convert to PyTorch Tensors
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Initialize Model
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f" Starting Training for {EPOCHS} epochs...")
    
    # Storage for all metrics
    history = {'acc': [], 'loss': [], 'time': []}

    for epoch in range(EPOCHS):
        start_time = time.time()  # <--- START TIMER
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        end_time = time.time()    # <--- STOP TIMER
        duration = end_time - start_time

        epoch_acc = 100 * correct / total
        
        # Save metrics to history
        history['acc'].append(epoch_acc)
        history['loss'].append(running_loss)
        history['time'].append(duration) # <--- Save Time
        
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Accuracy: {epoch_acc:.2f}% | Time: {duration:.2f}s")

    # --- EVALUATION ---
    print("\n📝 Evaluating on Test Data...")
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_acc = 100 * correct / total
    print(f" Final Test Accuracy: {final_acc:.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), "drought_model.pth")
    print("Model saved as 'drought_model.pth'")
    
    # --- GRAPH GENERATION (UPDATED) ---
    print("\n Generating Performance Graphs...")
    
    # Create a figure with THREE subplots (Accuracy, Loss, Time)
    plt.figure(figsize=(18, 5)) # Wider figure to fit 3 graphs

    # Graph 1: Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(range(1, EPOCHS + 1), history['acc'], label='Training Accuracy', color='green', marker='o')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Graph 2: Loss
    plt.subplot(1, 3, 2)
    plt.plot(range(1, EPOCHS + 1), history['loss'], label='Training Loss', color='red', marker='x', linestyle='--')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Graph 3: Time (NEW)     plt.subplot(1, 3, 3)
    plt.plot(range(1, EPOCHS + 1), history['time'], label='Time per Epoch', color='purple', marker='s')
    plt.title('Training Speed (Time)')
    plt.xlabel('Epochs')
    plt.ylabel('Seconds')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Layout adjustment
    plt.tight_layout()

    # Save the graph to a file
    plt.savefig("performance_graph.png")
    print(" Graph saved as 'performance_graph.png'. Check your folder!")

    # Save history numbers just in case
    np.save("training_history.npy", history)
    
    plt.show()