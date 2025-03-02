#!/usr/bin/env python3
"""
Qwen2.5 VL Model API Server for Render Deployment
This server loads the Qwen2.5-VL-7B-Instruct model and exposes endpoints to analyze images.
"""

import os
import base64
import json
import logging
import time
import io
from typing import Dict, List, Optional, Union, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
from PIL import Image
from transformers import AutoProcessor, TextIteratorStreamer, BitsAndBytesConfig
# Import the specific model class for Qwen2.5-VL
try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    # Fallback to AutoModelForCausalLM if the specific class isn't available
    from transformers import AutoModelForCausalLM
    Qwen2VLForConditionalGeneration = AutoModelForCausalLM
import uvicorn
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QwenVL-API")

# Model configuration
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
DEVICE = os.environ.get("DEVICE", "cpu")
USE_INT8 = os.environ.get("USE_INT8", "true").lower() == "true"
USE_FLOAT16 = os.environ.get("USE_FLOAT16", "false").lower() == "true"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))

# Initialize FastAPI app
app = FastAPI(
    title="Qwen VL API",
    description="API for the Qwen2.5-VL-7B-Instruct model to analyze images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processor
model = None
processor = None

# Input/Output models
class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="Content of the message. Can be a string or a list of content parts with different types.")

class AnalyzeImageRequest(BaseModel):
    messages: List[Message] = Field(..., description="Conversation messages including system prompt, user message with image, etc.")
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL to an image")
    max_new_tokens: Optional[int] = Field(MAX_NEW_TOKENS, description="Maximum number of tokens to generate")
    include_json: bool = Field(False, description="Whether to parse and return structured JSON from the response")

class AnalyzeScreenRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    parsed_content: List[Dict[str, Any]] = Field(..., description="Parsed screen elements from OmniParser")
    task_description: str = Field(..., description="Description of the analysis task")
    max_new_tokens: Optional[int] = Field(MAX_NEW_TOKENS, description="Maximum number of tokens to generate")

def load_model():
    """Load the Qwen VL model and processor"""
    global model, processor
    
    try:
        logger.info(f"Loading Qwen VL model from {MODEL_ID} on {DEVICE}")
        
        # Load processor first
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        logger.info("Processor loaded successfully")
        
        # Most basic, reliable way to load Qwen models
        logger.info("Loading model with AutoModelForCausalLM")
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,  # Important for Qwen models
            device_map="auto"        # Let transformers decide the best way to map
        )
        
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Allow server to start anyway so we can investigate/debug through API
        return False

def process_image(image_data=None, image_url=None):
    """Process image from base64 data or URL"""
    try:
        if image_data:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert("RGB")
        elif image_url:
            # Fetch image from URL
            import requests
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                return image.convert("RGB")
            else:
                raise ValueError(f"Failed to fetch image from URL: {response.status_code}")
        else:
            raise ValueError("No image data or URL provided")
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise ValueError(f"Error processing image: {str(e)}")

def format_parsed_content(parsed_content):
    """Format parsed content for the model prompt"""
    result = ""
    for i, element in enumerate(parsed_content[:100]):  # Limit to first 100 elements
        element_type = element.get("type", "unknown")
        content = element.get("content", "")
        box = element.get("box", [0, 0, 0, 0])
        
        if content:
            result += f"Element {i+1}: Type={element_type}, Content={content}, Position={box}\n"
    
    return result

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the server starts"""
    background_tasks = BackgroundTasks()
    background_tasks.add_task(load_model)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Qwen2.5-VL-7B-Instruct API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        # Model failed to load but server is running
        return JSONResponse(
            status_code=503,  # Service Unavailable
            content={
                "status": "unavailable", 
                "message": "Model not loaded. Check logs for details.",
                "model_id": MODEL_ID,
                "processor_loaded": processor is not None
            }
        )
    else:
        # Everything is working
        return JSONResponse(
            content={
                "status": "healthy",
                "model_id": MODEL_ID,
                "device": DEVICE
            }
        )

@app.post("/analyze")
async def analyze_image(request: AnalyzeImageRequest):
    """Analyze an image using Qwen VL model"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check the service logs."
        )
    
    try:
        # Get image from base64 data or URL
        if request.image_data:
            image = process_image(image_data=request.image_data)
        elif request.image_url:
            image = process_image(image_url=request.image_url)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either image_data or image_url must be provided"
            )
        
        start_time = time.time()
        
        # Prepare messages for the model
        messages = []
        for message in request.messages:
            if isinstance(message.content, str):
                messages.append({"role": message.role, "content": message.content})
            else:
                content_list = []
                for content in message.content:
                    if content.get("type") == "text":
                        content_list.append({"type": "text", "text": content.get("text", "")})
                    elif content.get("type") == "image" and image is not None:
                        content_list.append({"type": "image", "image": image})
                messages.append({"role": message.role, "content": content_list})
        
        # Process messages for the model
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Handle images in messages
        images = []
        for message in messages:
            if isinstance(message["content"], list):
                for content in message["content"]:
                    if content.get("type") == "image" and "image" in content:
                        images.append(content["image"])
        
        # Create model inputs
        inputs = processor(
            text=prompt, 
            images=images if images else None, 
            padding=True, 
            return_tensors="pt"
        ).to(DEVICE)
        
        # Generate response
        output = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=False
        )
        
        # Decode response
        response = processor.batch_decode(output, skip_special_tokens=True)[0]
        response = response.split("ASSISTANT: ")[-1].strip()
        
        # Try to extract JSON if requested
        extracted_json = None
        if request.include_json:
            try:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    extracted_json = json.loads(json_str)
            except:
                pass
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "model_id": MODEL_ID,
            "response": response,
            "json_data": extracted_json,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.post("/analyze_screen")
async def analyze_screen(request: AnalyzeScreenRequest):
    """Analyze a screen with parsed elements"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check the service logs."
        )
    
    try:
        # Process image
        image = process_image(image_data=request.image_data)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        start_time = time.time()
        
        # Format parsed content
        parsed_content_text = format_parsed_content(request.parsed_content)
        
        # Create system prompt
        system_prompt = """You are VerifiedX, an AI assistant specialized in analyzing screens and extracting content.
Your task is to analyze the screen image and the parsed elements to determine the best action to take.
Focus on identifying relevant UI elements and extracting text content that needs to be analyzed."""
        
        # Create user prompt
        user_prompt = f"""Task: {request.task_description}

Parsed screen elements:
{parsed_content_text}

Analyze this screen and determine the best action to take. Return a JSON with:
- action_type: The type of action to take (e.g., "extract_text", "click", "scroll")
- target_element_ids: IDs of elements to interact with or extract text from
- reasoning: Brief explanation of your recommendation
"""
        
        # Create messages for the model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        
        # Process messages for the model
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Create model inputs
        inputs = processor(
            text=prompt, 
            images=[image], 
            padding=True, 
            return_tensors="pt"
        ).to(DEVICE)
        
        # Generate response
        output = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=False
        )
        
        # Decode response
        response = processor.batch_decode(output, skip_special_tokens=True)[0]
        response = response.split("ASSISTANT: ")[-1].strip()
        
        # Try to extract JSON
        json_data = None
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                json_data = json.loads(json_str)
        except:
            pass
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "model_id": MODEL_ID,
            "analysis": response,
            "json_data": json_data,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error analyzing screen: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing screen: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    
    # Load the model before starting the server
    load_model()
    
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False) 