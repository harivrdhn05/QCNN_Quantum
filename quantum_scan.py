import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import torch
import time

# --- ADD THIS CONFIGURATION SECTION ---
N_WIRES = 4    
STRIDE = 2

dev = qml.device("default.qubit", wires=N_WIRES)

@qml.qnode(dev)
def quantum_circuit(inputs):
    # Encoding: Map 4 pixels to 4 qubits
    # We use RY rotations (Angle Embedding)
    for i in range(N_WIRES):
        qml.RY(inputs[i] * np.pi, wires=i)

    # Entanglement: Mix the information
    # This CNOT ring creates the "quantum correlations"
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])

    # Measurement: We extract 4 features
    return [qml.expval(qml.PauliZ(i)) for i in range(N_WIRES)]

def scan_image(image):
    """
    Slides the 2x2 quantum window over the image.
    Input: 64x64 image
    Output: 32x32x4 quantum feature map
    """
    # Image shape is (64, 64, 3) -> we only use 1 channel (average) for speed
    # Or just pick the Red channel (index 0) which is sensitive to drought
    if len(image.shape) == 3:
        image = image[:, :, 0] # Take Red Channel only
        
    h, w = image.shape
    new_h = h // STRIDE
    new_w = w // STRIDE
    
    # Pre-allocate output array (32x32 with 4 channels)
    out_image = np.zeros((new_h, new_w, 4))

    # The Sliding Window Loop
    for j in range(0, h, STRIDE):
        for k in range(0, w, STRIDE):
            # Get 2x2 pixels
            patch = [
                image[j, k],
                image[j, k+1],
                image[j+1, k],
                image[j+1, k+1]
            ]
            # Run Quantum Circuit
            q_results = quantum_circuit(patch)
            
            # Save results
            out_image[j//STRIDE, k//STRIDE] = q_results
            
    return out_image

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(" Starting Quantum Feature Extraction...")
    
    # Load Data
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    
    print(f"   Processing {len(X_train)} training images...")
    
    # Process Train Set
    X_train_q = []
    start_time = time.time()
    for i, img in enumerate(X_train):
        q_feat = scan_image(img)
        X_train_q.append(q_feat)
        if i % 10 == 0:
            print(f"      Scanned {i}/{len(X_train)} images...")
            
    # Process Test Set
    print(f"   Processing {len(X_test)} test images...")
    X_test_q = []
    for i, img in enumerate(X_test):
        q_feat = scan_image(img)
        X_test_q.append(q_feat)

    # Convert to NumPy and Save
    X_train_q = np.array(X_train_q)
    X_test_q = np.array(X_test_q)
    
    # Save as NEW files
    np.save("X_train_quantum.npy", X_train_q)
    np.save("X_test_quantum.npy", X_test_q)
    
    end_time = time.time()
    print(f"\n DONE! Process took {(end_time - start_time)/60:.1f} minutes.")
    print(f"   New Quantum Data Shape: {X_train_q.shape}")
    print("   (Images, 32, 32, 4 channels)")