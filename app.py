# https://huggingface.co/CompVis/stable-diffusion-v1-4

import math
import datetime
import time
import hashlib
import random

from flask import Flask, render_template, request
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda"

app = Flask(__name__)


def dummy_checker(images, **kwargs):
    return images, False


scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
)
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, use_auth_token=True
)
pipe.safety_checker = dummy_checker
pipe = pipe.to(device)
generator = torch.Generator("cuda")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.form["prompt"]
    steps = int(request.form.get("steps", 50))
    guidance_scale = float(request.form.get("guidance_scale", 7))
    seed = int(request.form.get("seed", random.random() * 1000000))
    width = int(request.form.get("width", 512))
    height = int(request.form.get("height", 512))
    amount = int(request.form.get("amount", 4))
    timestamp = math.floor(time.mktime(datetime.datetime.now().timetuple()))
    prompt_hash = hashlib.md5(bytearray("hello", encoding="utf-8")).hexdigest()
    output_prefix = f"static/out/{timestamp}-{prompt_hash}"
    print(output_prefix)

    images = []
    with autocast("cuda"):
        generator.manual_seed(seed)
        images = pipe(
            prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            width=width,
            height=height,
            num_images_per_prompt=amount
        ).images

    filenames = []
    for i in range(amount):
        filename = f"{output_prefix}-{i}.png"
        filenames.append(filename)
        images[i].save(filename)

    static_urls = [f"/{filename}" for filename in filenames]
    print(static_urls)

    return render_template("generate.html", prompt=prompt, images=static_urls)


if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")
