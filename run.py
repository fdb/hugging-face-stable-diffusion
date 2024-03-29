#!/bin/env python
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda"


def dummy_checker(images, **kwargs):
    return images, False


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe.safety_checker = dummy_checker
pipe = pipe.to(device)

prompt = "Swedish hand-made illustration with nature, lakes and woodlands,, retro illustration"
with autocast("cuda"):
    images = pipe(prompt, guidance_scale=7.5, num_images_per_prompt=4).images
    # image2 = pipe(prompt, guidance_scale=7.5).images[0]
    # image3 = pipe(prompt, guidance_scale=7.5).images[0]
    # image4 = pipe(prompt, guidance_scale=7.5).images[0]
for i, image in enumerate(images):
    image.save(f"output-{i+1}.png")

# image1.save("output-1.png")
# image2.save("output-2.png")
# image3.save("output-3.png")
# image4.save("output-4.png")
