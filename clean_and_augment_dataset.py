import os
import cv2
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 🔹 CONFIGURATION 🔹
INPUT_FOLDER = "processed_dataset"
OUTPUT_FOLDER = "cleaned_augmented_dataset"
IMAGE_SIZE = (32, 32)
AUGMENTATIONS_PER_IMAGE = 3  # How many times to augment each image

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 🔹 Transformations 🔹
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.Resize(IMAGE_SIZE)
])

# 🔹 Cleaning Function 🔹
def is_valid_image(file_path):
    try:
        img = Image.open(file_path).convert("RGB")
        img.verify()  # Will raise an exception if not valid
        return True
    except Exception as e:
        print(f"❌ Skipping corrupt file: {file_path} — {e}")
        return False

# 🔹 Augmentation and Save 🔹
def augment_and_save(image, base_name, count):
    for i in range(AUGMENTATIONS_PER_IMAGE):
        aug_image = augment_transform(image)
        save_name = f"{base_name}_aug{i+1}.png"
        aug_image.save(os.path.join(OUTPUT_FOLDER, save_name))

# 🔹 Main Process 🔹
print("🚀 Starting dataset cleaning and augmentation...")

for filename in tqdm(os.listdir(INPUT_FOLDER)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    file_path = os.path.join(INPUT_FOLDER, filename)

    if is_valid_image(file_path):
        try:
            img = Image.open(file_path).convert("RGB")
            img_resized = img.resize(IMAGE_SIZE, Image.NEAREST)

            # Save original resized version
            base_name = os.path.splitext(filename)[0]
            img_resized.save(os.path.join(OUTPUT_FOLDER, f"{base_name}.png"))

            # Apply augmentations
            augment_and_save(img_resized, base_name, AUGMENTATIONS_PER_IMAGE)

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

print("✅ Dataset cleaning and augmentation complete!")
