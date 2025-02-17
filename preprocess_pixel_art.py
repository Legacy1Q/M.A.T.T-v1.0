import os
import cv2
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# Set paths
DATASET_FOLDER = "path/to/your/dataset"  # Change this
OUTPUT_FOLDER = "processed_dataset"
IMAGE_SIZE = (32, 32)  # Adjust if needed
LABELS_FILE = "path/to/labels.json"  # Change if using a CSV

# Create output folder if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Load labels if available
def load_labels():
    if LABELS_FILE.endswith(".json"):
        with open(LABELS_FILE, "r") as f:
            return json.load(f)
    elif LABELS_FILE.endswith(".csv"):
        return pd.read_csv(LABELS_FILE).set_index("filename").to_dict()["label"]
    return {}


labels = load_labels()

# Process images
for filename in tqdm(os.listdir(DATASET_FOLDER)):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(DATASET_FOLDER, filename)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Keep transparency if present
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)  # Preserve pixel art sharpness
        image = image.astype(np.float32) / 255.0  # Normalize pixel values (0 to 1)

        # Save processed image
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(save_path, (image * 255).astype(np.uint8))

print("Processing complete! Images saved in", OUTPUT_FOLDER)
