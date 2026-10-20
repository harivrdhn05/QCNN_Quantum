import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# --- CONFIG ---
DATA_DIR = "dataset"
IMG_SIZE = 64
VALID_CLASSES = ["drought", "healthy"] # These are Class 1 (Valid)
INVALID_CLASS = "random"               # This is Class 0 (Invalid)

def load_data():
    data = []
    labels = []
    
    # Load Valid Satellite Images (Label 1)
    print("Loading Valid Satellite Images...")
    for category in VALID_CLASSES:
        path = os.path.join(DATA_DIR, category)
        if os.path.exists(path):
            for img_name in os.listdir(path)[:100]: # Limit to 100 each to balance
                try:
                    img_path = os.path.join(path, img_name)
                    img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                    data.append(np.array(img))
                    labels.append(1) # 1 = VALID SATELLITE
                except: pass

    # Load Random Images (Label 0)
    print("Loading Random 'Junk' Images...")
    path = os.path.join(DATA_DIR, INVALID_CLASS)
    if os.path.exists(path):
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                data.append(np.array(img))
                labels.append(0) # 0 = INVALID
            except: pass
    else:
        print(f"❌ Error: Please create a '{INVALID_CLASS}' folder in 'dataset' and add random photos.")
        exit()

    X = np.array(data) / 255.0
    y = np.array(labels)
    return X, y

# Simple CNN for Gatekeeping
class Gatekeeper(nn.Module):
    def __init__(self):
        super(Gatekeeper, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    X, y = load_data()
    
    # Convert to Tensors
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y).long()
    
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = Gatekeeper()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(" Training Gatekeeper...")
    for epoch in range(15):
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), "gatekeeper.pth")
    print(" Gatekeeper trained and saved as 'gatekeeper.pth'")