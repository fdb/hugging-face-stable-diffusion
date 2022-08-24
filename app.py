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

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

app = Flask(__name__)

def dummy_checker(images, **kwargs):
    return images, False

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, use_auth_token=True)
pipe.safety_checker = dummy_checker
pipe = pipe.to(device)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form["prompt"]
    steps = int(request.form.get("steps", 50))
    guidance_scale = float(request.form.get("guidance_scale", 7.5))
    seed = int(request.form.get("seed", random.random() * 1000000))
    print("steps",steps)
    timestamp = math.floor(time.mktime(datetime.datetime.now().timetuple()))
    prompt_hash = hashlib.md5(bytearray('hello', encoding='utf-8')).hexdigest()
    output_prefix = f"static/out/{timestamp}-{prompt_hash}"
    print(output_prefix)

    generator1 = torch.Generator("cuda").manual_seed(seed * 100 + 0)
    generator2 = torch.Generator("cuda").manual_seed(seed * 100 + 1)
    generator3 = torch.Generator("cuda").manual_seed(seed * 100 + 2)
    generator4 = torch.Generator("cuda").manual_seed(seed * 100 + 3)

    with autocast("cuda"):
        image1 = pipe(prompt, generator=generator1, guidance_scale=guidance_scale, num_inference_steps=steps)["sample"][0]
        image2 = pipe(prompt, generator=generator2, guidance_scale=guidance_scale, num_inference_steps=steps)["sample"][0]
        image3 = pipe(prompt, generator=generator3, guidance_scale=guidance_scale, num_inference_steps=steps)["sample"][0]
        image4 = pipe(prompt, generator=generator4, guidance_scale=guidance_scale, num_inference_steps=steps)["sample"][0]
    image1_filename = output_prefix + "-1.png"
    image2_filename = output_prefix + "-2.png"
    image3_filename = output_prefix + "-3.png"
    image4_filename = output_prefix + "-4.png"

    image1.save(image1_filename)
    image2.save(image2_filename)
    image3.save(image3_filename)
    image4.save(image4_filename)

    images = [image1_filename, image2_filename,image3_filename, image4_filename]
    # images = [image1_filename]
    images = [f"/{filename}" for filename in images]
    
    return render_template("generate.html", prompt=prompt, images=images)




if __name__=='__main__':
    app.run(debug=True, port=8000, host='0.0.0.0') 
