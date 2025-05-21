import os
import sys
import random
import torch
import torch.nn as nn
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# 1️⃣ Preprocessing: Load & Resize Dataset
DATASET_FOLDER = "processed_dataset"
IMAGE_SIZE = (32, 32)
SUBSET_SIZE = 50000  # Limit dataset size

all_images = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".png")]
SUBSET_SIZE = min(SUBSET_SIZE, len(all_images))
subset_images = random.sample(all_images, SUBSET_SIZE)  # Select images randomly


# Save subset file names for reference
with open("subset_images.txt", "w") as f:
    for img in subset_images:
        f.write(img + "\n")

print(f"✅ Selected {SUBSET_SIZE} images for training!")

# Define transformation pipeline
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1] range
])

class PixelArtDataset(Dataset):
    def __init__(self, dataset_folder, subset_file="subset_images.txt", prompt="pixel art"):
        # Load only the images from the subset file
        with open(subset_file, "r") as f:
            self.file_paths = [os.path.join(dataset_folder, line.strip()) for line in f.readlines()]
        self.prompt = prompt

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        image = transform(image)
        input_ids = tokenizer(self.prompt, return_tensors="pt").input_ids
        return {"pixel_values": image, "input_ids": input_ids}

# Load dataset
train_dataset = PixelArtDataset(DATASET_FOLDER)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# 2️⃣ Fine-Tuning: Train Stable Diffusion on Pixel Art
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to("cuda")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")
text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to("cuda")
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")

learning_rate = 5e-6
epochs = 3
optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

# 🔥 Check CLIP text encoder output size (either 512 or 768)
embed_dim = text_encoder.config.hidden_size

# 🔥 Define projection layer dynamically
text_proj = nn.Linear(embed_dim, 320).to("cuda")

for epoch in range(epochs):
    torch.cuda.empty_cache()
    print(f"Epoch {epoch + 1}/{epochs}...")
    for batch in train_dataloader:
        optimizer.zero_grad()
        pixel_values = batch["pixel_values"].to("cuda")
        input_ids = batch["input_ids"].to("cuda")

        with torch.no_grad():
            # 🔥 Generate text embeddings
            text_input = tokenizer(["pixel art"] * pixel_values.shape[0], padding="max_length", max_length=77,
                                   return_tensors="pt").to("cuda")
            text_embeddings = text_encoder(text_input.input_ids).last_hidden_state.to("cuda")

        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215  # Normalize latents for SD
        latents = latents.unsqueeze(0) if latents.dim() == 3 else latents  # Ensure batch dim exists

        timestep = torch.randint(0, 1000, (1,), device="cuda").long()  # Single integer tensor

        # 🔥 Pass correctly shaped embeddings to UNet
        noise_pred = unet(latents, timestep=timestep, encoder_hidden_states=text_embeddings).sample

        loss = torch.nn.functional.mse_loss(noise_pred, latents)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        optimizer.step()

    print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")

# 🔥 Save Fine-Tuned Model Correctly
unet.save_pretrained("./fine_tuned_pixel_art_model/unet")
vae.save_pretrained("./fine_tuned_pixel_art_model/vae")

print("✅ Fine-tuning complete! Model saved.")

# 3️⃣ Generate New Pixel Art from Text Prompts
fine_tuned_pipe = StableDiffusionPipeline.from_pretrained(
    model_id, unet=UNet2DConditionModel.from_pretrained("./fine_tuned_pixel_art_model/unet")
).to("cuda")

prompt = "32-bit warrior with sword"
image = fine_tuned_pipe(prompt).images[0]
image.show()
