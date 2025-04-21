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
import uuid

app = FastAPI()

# ✅ Make sure outputs folder exists before mounting
os.makedirs("outputs", exist_ok=True)

# Serve images from the 'outputs' directory
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load small & optimized model
model_id = "stabilityai/sd-turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    revision="fp16" if device == "cuda" else None,
    low_cpu_mem_usage=True
)
pipe.to(device)

# Define prompt schema
class PromptRequest(BaseModel):
    prompt: str

# POST /generate
@app.post("/generate")
def generate_image(data: PromptRequest):
    try:
        result = pipe(data.prompt, num_inference_steps=10, guidance_scale=2.0)  # turbo = lower steps
        image = result.images[0]

        # Save to 'outputs' folder
        os.makedirs("outputs", exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        path = os.path.join("outputs", filename)
        image.save(path)

        # 🔥 Update: Use Railway dynamic host instead of localhost
        base_url = os.getenv("RAILWAY_STATIC_URL", "http://localhost:8000")
        return {"image_url": f"{base_url}/outputs/{filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
