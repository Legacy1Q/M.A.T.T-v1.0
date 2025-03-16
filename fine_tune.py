import os
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader
from preprocess_pixel_art import train_dataset  # 🔹 NEW: Import dataset from preprocessing script

# 🔹 Load dataset with new anime pixel art data
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 🔹 Stable Diffusion Model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to("cuda")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# 🔹 Training Configuration
learning_rate = 5e-6
epochs = 3
optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

# 🔹 CLIP text encoder projection
embed_dim = text_encoder.config.hidden_size
text_proj = nn.Linear(embed_dim, 320).to("cuda")

# 🔹 Training Loop
for epoch in range(epochs):
    torch.cuda.empty_cache()
    print(f"Epoch {epoch + 1}/{epochs}...")

    for batch in train_dataloader:
        optimizer.zero_grad()
        pixel_values = batch["pixel_values"].to("cuda")
        prompt = batch["prompt"]

        with torch.no_grad():
            text_input = tokenizer([prompt] * pixel_values.shape[0], padding="max_length", max_length=77,
                                   return_tensors="pt").to("cuda")
            text_embeddings = text_encoder(text_input.input_ids).last_hidden_state.to("cuda")

        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215  # Normalize latents for SD
        latents = latents.unsqueeze(0) if latents.dim() == 3 else latents  # Ensure batch dim exists

        timestep = torch.randint(0, 1000, (1,), device="cuda").long()  # Single integer tensor
        noise_pred = unet(latents, timestep=timestep, encoder_hidden_states=text_embeddings).sample

        loss = torch.nn.functional.mse_loss(noise_pred, latents)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        optimizer.step()

    print(f"✅ Epoch {epoch + 1} completed. Loss: {loss.item()}")

# 🔹 Save Fine-Tuned Model
unet.save_pretrained("./fine_tuned_anime_pixel_art/unet")
vae.save_pretrained("./fine_tuned_anime_pixel_art/vae")

print("✅ Fine-tuning complete! Model saved.")
