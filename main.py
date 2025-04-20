from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO
from PIL import Image
from fastapi.staticfiles import StaticFiles
import os
import uuid  # ✅ Moved up here

app = FastAPI()

# Serve images from the 'outputs' directory
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Allow frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe.to(device)

# Prompt request format
class PromptRequest(BaseModel):
    prompt: str

# POST /generate endpoint
@app.post("/generate")
def generate_image(data: PromptRequest):
    try:
        result = pipe(data.prompt, num_inference_steps=30, guidance_scale=7.5)
        image = result.images[0]

        # Make sure 'outputs' directory exists
        os.makedirs("outputs", exist_ok=True)

        # Save image with a unique name
        image_filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join("outputs", image_filename)
        image.save(image_path)

        # Return public URL to the image
        return {"image_url": f"http://127.0.0.1:8000/outputs/{image_filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
