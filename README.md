# Small web app around Hugging Face's Stable Diffusion

## Setup

```
virtualenv --system-site-packages venv
source venv/bin/activate
pip install transformers huggingface diffusers scipy flask
```

Make a token at Hugging Face: https://huggingface.co/settings/tokens

Login through the command-line:

```
huggingface-cli login
```

Paste the token when asked

## Running the webserver

```
python app.py
```

Doing this the first time will download the model. Please be patient.
