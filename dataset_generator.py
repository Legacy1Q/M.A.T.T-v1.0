import os
import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO

# 🔹 CONFIGURATION SETTINGS 🔹
INPUT_FOLDER = "high_res_anime_images"  # Folder with HD images
OUTPUT_FOLDER = "processed_anime_pixel_art"  # Folder for pixelated images
IMAGE_SIZE = (32, 32)  # Target pixel art size (adjustable)
PIXEL_SCALE = 4  # Higher = more pixelated (e.g., 4 → blocky, 2 → less pixelated)
URL_LIST_FILE = "image_urls.txt"  # Text file with image URLs (optional)

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def download_images_from_urls(url_file, save_folder):
    """Downloads images from a text file containing image URLs."""
    if not os.path.exists(url_file):
        print("❌ No URL list found. Skipping downloads.")
        return

    with open(url_file, "r") as file:
        urls = file.readlines()

    for idx, url in enumerate(urls):
        url = url.strip()
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img_path = os.path.join(save_folder, f"downloaded_{idx}.png")
                img.save(img_path)
                print(f"✅ Downloaded: {img_path}")
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")

def convert_to_pixel_art(image_path, output_path, scale=PIXEL_SCALE):
    """Converts an image into pixel art by downscaling & upscaling."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to smaller resolution (downscale)
    small = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_LINEAR)

    # Resize back to original size (upscale)
    pixelated = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert back to PIL image and save
    pixelated_img = Image.fromarray(pixelated)
    pixelated_img = pixelated_img.resize(IMAGE_SIZE, Image.NEAREST)  # Resize to 32x32
    pixelated_img.save(output_path)

    print(f"🎨 Pixelated: {output_path}")

def process_images():
    """Processes all images in the input folder and converts them to pixel art."""
    images = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print("❌ No images found in input folder!")
        return

    for img_name in images:
        input_path = os.path.join(INPUT_FOLDER, img_name)
        output_path = os.path.join(OUTPUT_FOLDER, img_name)
        convert_to_pixel_art(input_path, output_path)

# 🔹 RUN SCRIPT 🔹
print("🚀 Starting Pixel Art Dataset Generation...")

# Step 1: Download images if URL list exists
download_images_from_urls(URL_LIST_FILE, INPUT_FOLDER)

# Step 2: Process images
process_images()

print("✅ All images processed successfully! Dataset ready.")
