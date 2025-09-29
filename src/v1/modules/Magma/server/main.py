import base64
import io
import os
from typing import List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Magma Model API",
    description="API for interacting with the Microsoft Magma multimodal model",
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Magma model
dtype = torch.bfloat16
model = None
processor = None

# Action processing parameters
n_action_bins = 256
bins = np.linspace(-1, 1, n_action_bins)
bin_centers = (bins[:-1] + bins[1:]) / 2.0

# Set up system message
system_message = {
    "role": "system",
    "content": "You are agent that can see, talk and act.",
}


def denormalize_actions(
    normalized_actions,
    action_low=np.array([-0.05, -0.05, -0.05, -3.14, -3.14, -3.14, 0]),
    action_high=np.array([0.05, 0.05, 0.05, 3.14, 3.14, 3.14, 1]),
):
    """Convert normalized actions (-1 to 1) to actual values using the specified range"""
    mask = np.ones_like(normalized_actions, dtype=bool)
    raw_action = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return raw_action


def generate_response(image, user_prompt):
    """Generate a response from the Magma model given an image and a prompt"""
    convs = [
        system_message,
        {"role": "user", "content": f"<image>\n{user_prompt}"},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )
    if model.config.mm_use_image_start_end:
        prompt = prompt.replace("<image>", "<image_start><image><image_end>")

    inputs = processor(images=[image], texts=prompt, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
    inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(0)
    inputs = inputs.to("cuda").to(dtype)

    generation_args = {
        "max_new_tokens": 500,
        "temperature": 0.7,  # Some temperature for diverse responses
        "do_sample": True,  # Enable sampling
        "num_beams": 1,
        "use_cache": True,
    }

    with torch.inference_mode():
        generate_ids = model.generate(**inputs, **generation_args)

    # For action IDs - extract the last 7 tokens (6 DOF + gripper)
    action_ids = generate_ids[0, -8:-1].cpu().tolist()

    # Convert to discretized actions
    discretized_actions = processor.tokenizer.vocab_size - np.array(action_ids).astype(
        np.int64
    )
    discretized_actions = np.clip(
        discretized_actions - 1, a_min=0, a_max=bin_centers.shape[0] - 1
    )
    normalized_actions = bin_centers[discretized_actions]

    # Convert normalized actions to actual delta values
    delta_values = denormalize_actions(normalized_actions)

    # For text response
    text_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
    response = processor.decode(text_ids[0], skip_special_tokens=True).strip()

    return normalized_actions.tolist(), delta_values.tolist(), response


@app.on_event("startup")
async def load_model():
    """Load the model when the application starts"""
    global model, processor
    print("Loading Magma model... This may take a while.")

    try:
        # Get model ID from environment or use default
        model_id = os.environ.get("MODEL_ID", "microsoft/Magma-8B")
        print(f"Loading model from: {model_id}")
        
        # Load the model from HuggingFace
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype
        )
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, use_fast=True
        )
        
        model.to("cuda")
        print("Magma model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        raise


class ImagePromptRequest(BaseModel):
    """Request model for sending both image and prompt"""
    image: str  # base64 encoded image
    prompt: str


class MagmaResponse(BaseModel):
    """Response model with text and actions"""
    text_response: str
    normalized_actions: List[float]
    delta_values: List[float]


@app.post("/predict", response_model=MagmaResponse)
async def predict(request: ImagePromptRequest):
    """Endpoint to make predictions using the Magma model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Generate the response
        normalized_actions, delta_values, text_response = generate_response(
            image, request.prompt
        )

        return {
            "text_response": text_response,
            "normalized_actions": normalized_actions,
            "delta_values": delta_values,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


@app.post("/predict_from_file", response_model=MagmaResponse)
async def predict_from_file(file: UploadFile = File(...), prompt: str = Form(...)):
    """Endpoint to make predictions using an uploaded file"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # Read the image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Generate the response
        normalized_actions, delta_values, text_response = generate_response(
            image, prompt
        )

        return {
            "text_response": text_response,
            "normalized_actions": normalized_actions,
            "delta_values": delta_values,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Endpoint to check if the API is running"""
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    # Run the server on 0.0.0.0 to accept connections from any IP address
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
