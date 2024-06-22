import os
import argparse
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import alpha_clip
from dataset.dataset import TrainDataset, ValidationDataset
from safetensors.torch import load_model

import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/training_config.yaml")
    return parser.parse_args()

@torch.inference_mode()
def calculate_validation_loss(val_dataloader, projection_layer, clip_model, clip_image_processor, alpha_clip_model, alpha_clip_preprocess, mask_transform, loss_fn, device):
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(val_dataloader, desc="Validation"):
        projection_layer.eval()
        image = batch["image"] 
        mask = batch["mask"]
        image_pil = [torchvision.transforms.ToPILImage()(img) for img in image]
        mask_pil = [torchvision.transforms.ToPILImage()(img) for img in mask]
        
        # Preprocess images for CLIP and alpha-CLIP
        preprocessed_image_for_clip = clip_image_processor(images=image_pil, return_tensors="pt").pixel_values
        clip_image_embeds = clip_model(preprocessed_image_for_clip.to(device, dtype=torch.float32)).image_embeds
        preprocessed_image_for_alpha_clip = preprocess_images(image_pil, alpha_clip_preprocess, device)
        preprocessed_mask = preprocess_masks(mask_pil, mask_transform, device)
        alpha_clip_image_embeds = alpha_clip_model.encode_image(preprocessed_image_for_alpha_clip, preprocessed_mask)
        
        # Project alpha-CLIP embeddings
        projected_alpha_clip_image_embeds = projection_layer(alpha_clip_image_embeds.to(device, dtype=torch.float32))
        
        # Compute loss
        loss_value = loss_fn(projected_alpha_clip_image_embeds, clip_image_embeds)
        total_loss += loss_value.item()
        total_samples += 1

    validation_loss = total_loss / total_samples
    return validation_loss

def generate_projection_layer(config, device="cuda"):
    projection_layer = nn.ModuleList()
    
    for i in range(config.number_of_hidden_layers):
        if i < config.number_of_hidden_layers // 2:
            projection_layer.append(nn.Linear(config.alpha_clip_embed_dim, config.alpha_clip_embed_dim))
            projection_layer.append(nn.LayerNorm(config.alpha_clip_embed_dim))
        elif i == config.number_of_hidden_layers // 2:
            projection_layer.append(nn.Linear(config.alpha_clip_embed_dim, config.ip_adapter_embed_dim))
            projection_layer.append(nn.LayerNorm(config.ip_adapter_embed_dim))
        else:
            projection_layer.append(nn.Linear(config.ip_adapter_embed_dim, config.ip_adapter_embed_dim))
            projection_layer.append(nn.LayerNorm(config.ip_adapter_embed_dim))
        projection_layer.append(nn.GELU())
        
    projection_layer.append(nn.Linear(config.ip_adapter_embed_dim, config.ip_adapter_embed_dim))

    return nn.Sequential(*projection_layer).to(device).to(torch.float32)

def preprocess_images(images, preprocess_fn, device):
    return torch.stack([preprocess_fn(img) for img in images]).to(device, dtype=torch.float32)

def preprocess_masks(masks, transform_fn, device):
    return torch.stack([transform_fn(mask) for mask in masks]).to(device, dtype=torch.float32)
            
def train(config, train_dataloader, val_dataloader, device="cuda"):
    accelerator = Accelerator()
    device = accelerator.device

    # Load and prepare models and processors
    clip_model = CLIPVisionModelWithProjection.from_pretrained(config.image_encoder_path).to(device).to(torch.float32)
    clip_image_processor = CLIPImageProcessor()
    alpha_clip_model, alpha_clip_preprocess = alpha_clip.load(config.alpha_clip_id, alpha_vision_ckpt_pth=config.alpha_vision_ckpt_pth, device=config.device)
    alpha_clip_model = alpha_clip_model.to(device).to(torch.float32)

    # Define mask transformation for alpha-CLIP
    if "@336" in config.alpha_clip_id:
        mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((336, 336)), # change to (336,336) when using ViT-L/14@336px
            transforms.Normalize(0.5, 0.26)
        ])
    else:
        mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)), # change to (336,336) when using ViT-L/14@336px
            transforms.Normalize(0.5, 0.26)
        ])

    # Generate and prepare projection layer
    projection_layer = generate_projection_layer(config, device)
    
    if config.mlp_projection_layer_ckpt_path:
        load_model(projection_layer, config.mlp_projection_layer_ckpt_path)

    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(projection_layer.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    iteration_count = 0
    projection_layer, optimizer, train_dataloader,val_dataloader = accelerator.prepare(projection_layer, optimizer, train_dataloader, val_dataloader)
        
    for _ in tqdm(range(config.epochs), desc="Epochs"):
        for batch in tqdm(train_dataloader, desc="Training"):
            projection_layer.train()
            image = batch["image"] 
            mask = batch["mask"]
            image_pil = [torchvision.transforms.ToPILImage()(img) for img in image]
            mask_pil = [torchvision.transforms.ToPILImage()(img) for img in mask]

            # Preprocess images for CLIP and alpha-CLIP
            preprocessed_image_for_clip = clip_image_processor(images=image_pil, return_tensors="pt").pixel_values
            clip_image_embeds = clip_model(preprocessed_image_for_clip.to(device, dtype=torch.float32)).image_embeds

            preprocessed_image_for_alpha_clip = preprocess_images(image_pil, alpha_clip_preprocess, device)
            preprocessed_mask = preprocess_masks(mask_pil, mask_transform, device)
            alpha_clip_image_embeds = alpha_clip_model.encode_image(preprocessed_image_for_alpha_clip, preprocessed_mask)

            # Project alpha-CLIP embeddings
            projected_alpha_clip_image_embeds = projection_layer(alpha_clip_image_embeds.to(device, dtype=torch.float32))

            # Compute loss and update weights
            optimizer.zero_grad()
            loss_value = loss_fn(projected_alpha_clip_image_embeds, clip_image_embeds)
            accelerator.backward(loss_value)
            optimizer.step()
            iteration_count += 1
            
            print(f"Iteration: {iteration_count}, Loss: {loss_value.item()}")
            
            # Call the function in the train loop
            if iteration_count % config.eval_interval == 0:
                projection_layer.eval()
                validation_loss = calculate_validation_loss(val_dataloader, projection_layer, clip_model, clip_image_processor, alpha_clip_model, alpha_clip_preprocess, mask_transform, loss_fn, device)
                print(f"Iteration: {iteration_count}, Validation Loss: {validation_loss}")
            
            # Save model checkpoints at specified intervals
            if iteration_count % config.save_interval == 0:
                accelerator.wait_for_everyone()
                accelerator.save_model(projection_layer, config.save_path)
                
def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    os.makedirs(config.save_path, exist_ok=True)
    train_dataset = TrainDataset(
        path=config.data_path
    )
    val_dataset = ValidationDataset(
        path=config.val_path
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.train_batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=config.val_batch_size)
    train(config, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()
