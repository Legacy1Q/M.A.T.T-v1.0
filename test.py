import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL

# ✅ Load fine-tuned UNet and VAE on CUDA
unet = UNet2DConditionModel.from_pretrained("./fine_tuned_pixel_art_model/unet").to("cuda")
vae = AutoencoderKL.from_pretrained("./fine_tuned_pixel_art_model/vae").to("cuda")

# ✅ Check if models are on CUDA
print(f"UNet on CUDA: {next(unet.parameters()).is_cuda}")  # Should print: True
print(f"VAE on CUDA: {next(vae.parameters()).is_cuda}")  # Should print: True

# Load the fine-tuned model
fine_tuned_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    unet=UNet2DConditionModel.from_pretrained("./fine_tuned_pixel_art_model/unet")
).to("cuda")

# Generate an image
prompt = "8-bit warrior with sword"
image = fine_tuned_pipe(prompt).images[0]

# ✅ Show the generated image
image.show()

# Optionally, save the image
image.save("generated_pixel_art.png")
