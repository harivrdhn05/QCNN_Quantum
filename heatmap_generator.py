import numpy as np
import pennylane as qml
import torch
import cv2
import os
from PIL import Image

# --- CONFIGURATION ---
N_WIRES = 4
STRIDE = 2  # Use 2 for speed, 1 for high detail
OUTPUT_DIR = "static" # Folder to save images
# ---------------------

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. DEFINE QUANTUM LAYER ---
dev = qml.device("default.qubit", wires=N_WIRES)

@qml.qnode(dev)
def quantum_circuit(inputs):
    # Encoding
    for i in range(N_WIRES):
        qml.RY(inputs[i] * np.pi, wires=i)
    # Entanglement
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])
    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(N_WIRES)]

def get_next_filename():
    """Finds the next available filename like heatmap_1.png, heatmap_2.png..."""
    i = 1
    while True:
        filename = f"heatmap_{i}.png"
        full_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(full_path):
            return full_path
        i += 1

def generate_heatmap(image_path):
    print(f"🔍 Generating Quantum Heatmap for: {image_path}")
    
    # 1. Load Image
    try:
        img = Image.open(image_path).convert('RGB').resize((64, 64))
        img_array = np.array(img) / 255.0
    except Exception as e:
        return None, str(e)

    # Use Red Channel (Band 0)
    data = img_array[:, :, 0] 
    h, w = data.shape
    
    # 2. Create Empty Heatmap Grid
    heatmap_grid = np.zeros((h, w))

    # 3. Sliding Window Scan
    for j in range(0, h-1, STRIDE):
        for k in range(0, w-1, STRIDE):
            patch = [
                data[j, k], data[j, k+1],
                data[j+1, k], data[j+1, k+1]
            ]
            q_results = quantum_circuit(patch)
            score = np.mean(q_results) 
            heatmap_grid[j:j+STRIDE, k:k+STRIDE] = score

    # 4. Normalize & Color Map
    heatmap_norm = cv2.normalize(heatmap_grid, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap_norm = np.uint8(heatmap_norm)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # 5. Overlay
    original_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(original_cv, 0.6, heatmap_color, 0.4, 0)
    
    # 6. Save with INCREMENTAL Name
    save_path = get_next_filename() # <--- NEW LOGIC HERE
    cv2.imwrite(save_path, superimposed)
    
    print(f"✅ Heatmap saved to {save_path}")
    
    return save_path, "Success"

if __name__ == "__main__":
    # Test
    generate_heatmap("dataset/drought/drought_2023-04-12_0.png")