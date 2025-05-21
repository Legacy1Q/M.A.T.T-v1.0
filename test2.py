import os
from PIL import Image
import matplotlib.pyplot as plt

# CONFIG
DATASET_FOLDER = "cleaned_augmented_dataset"
GRID_SIZE = 5  # 5x5 = 25 images
IMAGE_SIZE = (32, 32)

# Load image paths
image_paths = [os.path.join(DATASET_FOLDER, f) for f in os.listdir(DATASET_FOLDER)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]

# Select N random images
image_paths = image_paths[:GRID_SIZE * GRID_SIZE]

# Plot them in a grid
plt.figure(figsize=(10, 10))
for i, img_path in enumerate(image_paths):
    img = Image.open(img_path)
    plt.subplot(GRID_SIZE, GRID_SIZE, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(os.path.basename(img_path), fontsize=6)
plt.tight_layout()
plt.show()

