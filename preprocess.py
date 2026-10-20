import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_DIR = "dataset"            # Your main folder
IMG_SIZE = 64                   # Must match the download size
CLASSES = ["drought", "healthy"] # Class 0 = Drought, Class 1 = Healthy
# ---------------------

def load_data():
    print("🔄 Loading images and converting to arrays...")
    
    data = []
    labels = []
    
    # Loop through both folders (drought/healthy)
    for category in CLASSES:
        path = os.path.join(DATA_DIR, category)
        class_num = CLASSES.index(category) # 0 or 1
        
        # Check if folder exists
        if not os.path.exists(path):
            print(f"⚠️ Warning: Folder '{path}' not found!")
            continue

        # Loop through every image in the folder
        count = 0
        for img_name in os.listdir(path):
            try:
                # 1. Open Image
                img_path = os.path.join(path, img_name)
                img = Image.open(img_path).convert('RGB') # Ensure 3 channels
                
                # 2. Resize (Just to be safe)
                img = img.resize((IMG_SIZE, IMG_SIZE))
                
                # 3. Convert to NumPy Array
                img_array = np.array(img)
                
                # 4. Append to list
                data.append(img_array)
                labels.append(class_num)
                count += 1
            except Exception as e:
                pass # Skip broken files

        print(f"   ✅ Loaded {count} images for class: {category}")

    # Convert lists to efficient NumPy arrays
    X = np.array(data)
    y = np.array(labels)
    
    # --- CRITICAL STEP: NORMALIZATION ---
    # We convert pixel values from [0, 255] to [0.0, 1.0]
    # This is required for Quantum Angle Encoding.
    X = X / 255.0
    
    return X, y

if __name__ == "__main__":
    # 1. Load Data
    X, y = load_data()
    
    if len(X) == 0:
        print("❌ Error: No data found. Did you download the images?")
        exit()

    print(f"\n📊 Total Dataset Shape: {X.shape}")
    print(f"   (Images, Height, Width, Channels)")
    
    # 2. Split into Train (80%) and Test (20%)
    # We shuffle automatically here so dry/wet are mixed.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Save to disk
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    
    print("\n💾 Success! Data saved as .npy files.")
    print(f"   Training Set: {len(X_train)} images")
    print(f"   Testing Set:  {len(X_test)} images")
    print("   Ready for Quantum Circuit building!")