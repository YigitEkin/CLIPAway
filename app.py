import gradio as gr
import sys
import torch
from omegaconf import OmegaConf
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from model.clip_away import CLIPAway
import cv2
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/inference_config.yaml", help="Path to the config file")
parser.add_argument("--share", action="store_true", help="Share the interface if provided")
args = parser.parse_args()

# Load configuration and models
config = OmegaConf.load(args.config)
sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", safety_checker=None, torch_dtype=torch.float32
)
clipaway = CLIPAway(
    sd_pipe=sd_pipeline, 
    image_encoder_path=config.image_encoder_path,
    ip_ckpt=config.ip_adapter_ckpt_path, 
    alpha_clip_path=config.alpha_clip_ckpt_pth, 
    config=config, 
    alpha_clip_id=config.alpha_clip_id, 
    device=config.device, 
    num_tokens=4
)

def dilate_mask(mask, kernel_size=5, iterations=5):
    mask = mask.convert("L")
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(np.array(mask), kernel, iterations=iterations)
    return Image.fromarray(mask)

def combine_masks(uploaded_mask, sketched_mask):
    if uploaded_mask is not None:
        return uploaded_mask
    elif sketched_mask is not None:
        return sketched_mask
    else:
        raise ValueError("Please provide a mask")

def remove_obj(image, uploaded_mask, seed):
    image_pil, sketched_mask = image["image"], image["mask"]
    mask = dilate_mask(combine_masks(uploaded_mask, sketched_mask))
    seed = int(seed)
    latents = torch.randn((1, 4, 64, 64), generator=torch.Generator().manual_seed(seed)).to("cuda")
    final_image = clipaway.generate(
        prompt=[""], scale=1, seed=seed,
        pil_image=[image_pil], alpha=[mask], strength=1, latents=latents
    )[0]
    return final_image

# Define example data
examples = [
    ["assets/gradio_examples/images/1.jpg", "assets/gradio_examples/masks/1.png", 42],
    ["assets/gradio_examples/images/2.jpg", "assets/gradio_examples/masks/2.png", 42],
    ["assets/gradio_examples/images/3.jpg", "assets/gradio_examples/masks/3.png", 2024],
]

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center'>CLIPAway: Harmonizing Focused Embeddings for Removing Objects via Diffusion Models</h1>")
    gr.Markdown("""
        <div style='display:flex; justify-content:center; align-items:center;'>
            <a href='https://arxiv.org/abs/2406.09368' style="margin:10px;">Paper</a> |
            <a href='https://yigitekin.github.io/CLIPAway/' style="margin:10px;">Project Website</a> |
            <a href='https://github.com/YigitEkin/CLIPAway' style="margin:10px;">GitHub</a>
        </div>
    """)
    gr.Markdown("""
            This application allows you to remove objects from images using the CLIPAway method with diffusion models.
            To use this tool:
            1. Upload an image.
            2. Either Sketch a mask over the object you want to remove or upload a pre-defined mask if you have one.
            4. Set the seed for reproducibility (default is 42).
            5. Click 'Remove Object' to process the image.
            6. The result will be displayed on the right side.
            Note: The mask should be a binary image where the object to be removed is white and the background is black.
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image and Sketch Mask", type="pil", tool="sketch")
            uploaded_mask = gr.Image(label="Upload Mask (Optional)", type="pil", optional=True)
            seed_input = gr.Number(value=42, label="Seed")
            process_button = gr.Button("Remove Object")
        with gr.Column():
            result_image = gr.Image(label="Result")
    
    process_button.click(
        fn=remove_obj,
        inputs=[image_input, uploaded_mask, seed_input],
        outputs=result_image
    )

    gr.Examples(
        examples=examples,
        inputs=[image_input, uploaded_mask, seed_input],
        outputs=result_image
    )

# Launch the interface with caching
if args.share:
    demo.launch(share=True)
else:
    demo.launch()
