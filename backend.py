import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import io
import os
import base64

# Import your new heatmap generator
from heatmap_generator import generate_heatmap

app = Flask(__name__)
CORS(app)

# Ensure 'static' folder exists for saving heatmaps
os.makedirs("static", exist_ok=True)

# --- 1. DEFINE GATEKEEPER ---
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

# --- 2. DEFINE MAIN MODEL ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 2)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        return x

# --- 3. LOAD MODELS ---
gatekeeper = Gatekeeper()
model = Net()

try:
    gatekeeper.load_state_dict(torch.load("gatekeeper.pth", map_location=torch.device('cpu')))
    gatekeeper.eval()
    print("✅ Gatekeeper loaded!")
except:
    print("⚠️ Warning: 'gatekeeper.pth' not found.")

try:
    model.load_state_dict(torch.load("drought_model.pth", map_location=torch.device('cpu')))
    model.eval()
    print("✅ Main Quantum Model loaded!")
except:
    print("⚠️ Error: 'drought_model.pth' not found.")

# --- 4. QUANTUM CIRCUIT SETUP (For Prediction) ---
dev = qml.device("default.qubit", wires=4)
@qml.qnode(dev)
def quantum_circuit(inputs):
    for i in range(4):
        qml.RY(inputs[i] * np.pi, wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

def scan_image(image_array):
    img = image_array[:, :, 0] # Use Red Channel
    h, w = img.shape
    new_h, new_w = h // 2, w // 2
    out_image = np.zeros((new_h, new_w, 4))
    for j in range(0, h, 2):
        for k in range(0, w, 2):
            patch = [img[j,k], img[j,k+1], img[j+1,k], img[j+1,k+1]]
            out_image[j//2, k//2] = quantum_circuit(patch)
    return out_image

# --- 5. API ENDPOINTS ---

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize((64, 64))
        img_array = np.array(image) / 255.0
        
        # Security Check
        check_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            security_check = gatekeeper(check_tensor)
            probs = torch.softmax(security_check, dim=1)
            is_satellite = probs[0][1].item()

        if is_satellite < 0.8: 
            return jsonify({
                'result': 'Error',
                'message': 'Invalid Image. Not a satellite map.',
                'confidence': 0
            })

        # Quantum Prediction
        q_features = scan_image(img_array)
        input_tensor = torch.tensor(q_features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            drought_score = probs[0][0].item()
            healthy_score = probs[0][1].item()
            
        result = "Drought" if drought_score > healthy_score else "Healthy"
        confidence = max(drought_score, healthy_score) * 100
        
        return jsonify({
            'result': result,
            'confidence': f"{confidence:.1f}%",
            'drought_prob': drought_score,
            'healthy_prob': healthy_score
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- NEW HEATMAP ENDPOINT ---
@app.route('/heatmap', methods=['POST'])
def get_heatmap():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # Save temp file
        temp_path = "static/temp_input.png"
        file.save(temp_path)
        
        # Generate Heatmap
        output_path, status = generate_heatmap(temp_path, "static/heatmap_result.png")
        
        if not output_path:
            return jsonify({'error': status}), 500
            
        # Return the image directly
        return send_file(output_path, mimetype='image/png')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)