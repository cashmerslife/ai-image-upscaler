import gradio as gr
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import numpy as np
import requests
import io
import time
import os

# Initialize the pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

try:
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True
    )
    pipe = pipe.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None

def upscale_image(
    image,
    prompt="",
    negative_prompt="",
    scale_factor=2,
    dynamic=6,
    creativity=0.35,
    resemblance=0.6,
    tiling_width=16,
    tiling_height=16,
    sd_model="stabilityai/stable-diffusion-xl-refiner-1.0",
    scheduler="DPM++ 2M Karras",
    num_inference_steps=18,
    seed=1337,
    downscaling=False,
    downscaling_resolution=768
):
    """
    Upscale an image using Stable Diffusion XL
    """
    if image is None:
        return None, "Please upload an image"
    
    if pipe is None:
        return None, "Model not loaded. Please check your setup."
    
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Get original dimensions
        original_width, original_height = image.size
        
        # Calculate new dimensions
        if downscaling and max(original_width, original_height) > downscaling_resolution:
            # Downscale first
            ratio = downscaling_resolution / max(original_width, original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Apply scale factor
        target_width = int(image.width * scale_factor)
        target_height = int(image.height * scale_factor)
        
        # Ensure dimensions are multiples of 8 (required by Stable Diffusion)
        target_width = (target_width // 8) * 8
        target_height = (target_height // 8) * 8
        
        # Resize input image to target size
        input_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Set up generation parameters
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Default prompt if none provided
        if not prompt.strip():
            prompt = "high quality, detailed, sharp, professional photography"
        
        # Default negative prompt if none provided
        if not negative_prompt.strip():
            negative_prompt = "blurry, low quality, distorted, artifacts, noise"
        
        # Generate the upscaled image
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=creativity,  # How much to change the image
            guidance_scale=dynamic,  # How closely to follow the prompt
            num_inference_steps=num_inference_steps,
            generator=generator
        )
        
        upscaled_image = result.images[0]
        
        return upscaled_image, f"Successfully upscaled from {image.width}x{image.height} to {upscaled_image.width}x{upscaled_image.height}"
        
    except Exception as e:
        return None, f"Error during upscaling: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="AI Image Upscaler") as demo:
    gr.Markdown("# 🚀 AI Image Upscaler")
    gr.Markdown("Upload an image and enhance it using Stable Diffusion XL for high-quality upscaling.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            input_image = gr.Image(
                label="Input Image",
                type="pil",
                sources=["upload", "clipboard"]
            )
            
            # Prompt inputs
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the desired output (optional)",
                lines=2
            )
            
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="What to avoid in the output (optional)",
                lines=2
            )
            
            # Scale factor
            scale_factor = gr.Slider(
                minimum=1,
                maximum=4,
                value=2,
                step=0.1,
                label="Scale Factor",
                info="How much to upscale the image"
            )
            
            # Advanced settings
            with gr.Accordion("Advanced Settings", open=False):
                dynamic = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=6,
                    step=0.1,
                    label="Guidance Scale",
                    info="Higher values follow prompt more closely"
                )
                
                creativity = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.35,
                    step=0.05,
                    label="Creativity",
                    info="How much to modify the original image"
                )
                
                resemblance = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.6,
                    step=0.05,
                    label="Resemblance",
                    info="How closely to match the original"
                )
                
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=18,
                    step=1,
                    label="Inference Steps",
                    info="More steps = better quality but slower"
                )
                
                seed = gr.Number(
                    label="Seed",
                    value=1337,
                    precision=0,
                    info="Random seed for reproducible results"
                )
                
                downscaling = gr.Checkbox(
                    label="Enable Downscaling",
                    value=False,
                    info="Reduce image size before upscaling for speed"
                )
                
                downscaling_resolution = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=768,
                    step=32,
                    label="Downscaling Resolution",
                    info="Maximum dimension for downscaling"
                )
            
            # Process button
            process_btn = gr.Button("🚀 Upscale Image", variant="primary")
        
        with gr.Column(scale=1):
            # Output section
            output_image = gr.Image(label="Upscaled Image")
            output_info = gr.Textbox(label="Process Info", interactive=False)
    
    # Event handlers
    process_btn.click(
        fn=upscale_image,
        inputs=[
            input_image,
            prompt,
            negative_prompt,
            scale_factor,
            dynamic,
            creativity,
            resemblance,
            gr.Number(value=16, visible=False),  # tiling_width
            gr.Number(value=16, visible=False),  # tiling_height
            gr.Textbox(value="stabilityai/stable-diffusion-xl-refiner-1.0", visible=False),  # sd_model
            gr.Textbox(value="DPM++ 2M Karras", visible=False),  # scheduler
            num_inference_steps,
            seed,
            downscaling,
            downscaling_resolution
        ],
        outputs=[output_image, output_info]
    )

if __name__ == "__main__":
    demo.launch(share=True)
