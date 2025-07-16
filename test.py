import torch
import os
import shutil
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from PIL import Image
from rembg import remove

# ✅ Load fine-tuned UNet and VAE on CUDA
unet = UNet2DConditionModel.from_pretrained("./fine_tuned_pixel_art_model/unet").to("cuda")
vae = AutoencoderKL.from_pretrained("./fine_tuned_pixel_art_model/vae").to("cuda")

# ✅ Load the full fine-tuned pipeline
fine_tuned_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    unet=unet,
    vae=vae
).to("cuda")

# ✅ Generate an image
prompt = "2D 32-Bit bullet"
generated_image = fine_tuned_pipe(prompt).images[0]

# ✅ Save the original image (with background)
original_path = "Assets/GeneratedSprites/generated_pixel_art.png"
generated_image.save(original_path, format="PNG")

# ✅ Remove background and save transparent image
with Image.open(original_path) as img:
    transparent_img = remove(img)
    transparent_path = "Assets/GeneratedSprites/generated_pixel_art_transparent.png"
    transparent_img.save(transparent_path)

# ✅ (Optional) Copy transparent image to Unity project
shutil.copy(
    transparent_path,
    "C:/Users/mdsim/Delivery Driver/Assets/GeneratedImages/my_sprite.png"
)

# ✅ (Optional) Preview
transparent_img.show()
