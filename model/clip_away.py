"""
modified from from https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py
"""
import os
from typing import List
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
import alpha_clip
from .utils import get_generator
from .attention_processor import AttnProcessor, IPAttnProcessor
from safetensors import safe_open
from safetensors.torch import load_model
import numpy as np

import torch.nn as nn


class ImageProjModel(torch.nn.Module):
    """Projection Model of IP-Adapter"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class CLIPAway:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, alpha_clip_path, config, alpha_clip_id="ViT-L/14", device="cuda", num_tokens=4):
        super().__init__()
        self.device = device
        self.ipadapter_image_encoder_path = image_encoder_path
        self.ipadapter_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()
        alpha_clip_model, alpha_clip_preprocess = alpha_clip.load(alpha_clip_id, alpha_vision_ckpt_pth=alpha_clip_path, device=device)

        # load image encoder
        self.image_encoder = alpha_clip_model.visual.to(self.device, dtype=torch.float32)
        
        self.clip_proj = CLIPVisionModelWithProjection.from_pretrained(self.ipadapter_image_encoder_path).to(
            self.device, dtype=torch.float32
        )
        self.alpha_clip_image_processor = alpha_clip_preprocess
        
        # preprocess mask transformation for alpha clip
        if "@336" in alpha_clip_id:
            self.mask_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((336, 336)), # change to (336,336) when using ViT-L/14@336px
                transforms.Normalize(0.5, 0.26)
            ])
        else:
            self.mask_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)), # change to (336,336) when using ViT-L/14@336px
                transforms.Normalize(0.5, 0.26)
            ])
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()
        self.mlp_projection_layer = self.generate_projection_layer(config)
        
        print(config.mlp_projection_layer_ckpt_path, type(config.mlp_projection_layer_ckpt_path) )
        if config.mlp_projection_layer_ckpt_path is not None:
            self.load_projection_layer(config.mlp_projection_layer_ckpt_path)
        
    def load_projection_layer(self, path):
        load_model(self.mlp_projection_layer, path)
        print("Projection layer loaded from", path)
        
    def generate_projection_layer(self, config):
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
        
        return nn.Sequential(*projection_layer).to(self.device).to(torch.float32)
    
    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.clip_proj.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float32)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor().to(self.device)
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float32)
        unet.set_attn_processor(attn_procs)
                
    def get_alpha_clip_embeds(self, pil_image, alpha):
        clip_image = [self.alpha_clip_image_processor(image) for image in pil_image]
        clip_image = torch.stack(clip_image).to(self.device, dtype=torch.float32)
        masks = [self.mask_transform(mask) for mask in alpha]
        masks = torch.stack(masks).to(self.device, dtype=torch.float32)
        
        return self.image_encoder(clip_image, masks)

    def load_ip_adapter(self):
        if os.path.splitext(self.ipadapter_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ipadapter_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ipadapter_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])
        
    def get_complement_of_mask(self, mask):
        return Image.fromarray((255 - np.array(mask[0])).astype(np.uint8))
    
    def clipaway_projection_block(self, bg_embeds, fg_embeds):
        projected_vector_magnitude = bg_embeds[0].dot(fg_embeds[0]) / fg_embeds[0].norm()
        projected_vector = projected_vector_magnitude * fg_embeds / fg_embeds.norm()
        return bg_embeds - projected_vector
    
    def get_focused_embeddings(self, pil_image, alpha, use_projection_block=False):
        # get focused alpha clip embeds
        clip_image_embeds_fg = self.get_alpha_clip_embeds(pil_image, alpha) 
        clip_image_embeds_bg = self.get_alpha_clip_embeds(pil_image, [self.get_complement_of_mask(alpha)])
        
        # mlp projection
        projected_alpha_clip_embeds_fg = self.mlp_projection_layer(clip_image_embeds_fg)
        projected_alpha_clip_embeds_bg = self.mlp_projection_layer(clip_image_embeds_bg)
            
        # ip adapter logic
        image_prompt_embeds_fg = self.image_proj_model(projected_alpha_clip_embeds_fg)
        image_prompt_embeds_bg = self.image_proj_model(projected_alpha_clip_embeds_bg)
        uncond_image_prompt_embeds = self.image_proj_model(self.mlp_projection_layer(torch.zeros_like(clip_image_embeds_fg)))
                
        if use_projection_block:
            # clipaway projection block
            projected_alpha_clip_embeds = self.clipaway_projection_block(projected_alpha_clip_embeds_bg, projected_alpha_clip_embeds_fg)
            image_prompt_embeds = self.image_proj_model(projected_alpha_clip_embeds)
            return image_prompt_embeds, image_prompt_embeds_fg, image_prompt_embeds_bg, uncond_image_prompt_embeds
        
        return image_prompt_embeds_fg, image_prompt_embeds_bg, uncond_image_prompt_embeds
        
        
    def get_ipadapter_embeds(self, pil_image=None, alpha=None):
        # get focused alpha clip embeds
        clip_image_embeds_fg = self.get_alpha_clip_embeds(pil_image, alpha) 
        clip_image_embeds_bg = self.get_alpha_clip_embeds(pil_image, [self.get_complement_of_mask(alpha)])
        
        # mlp projection
        projected_alpha_clip_embeds_fg = self.mlp_projection_layer(clip_image_embeds_fg)
        projected_alpha_clip_embeds_bg = self.mlp_projection_layer(clip_image_embeds_bg)
        
        # clipaway projection block
        projected_alpha_clip_embeds = self.clipaway_projection_block(projected_alpha_clip_embeds_bg, projected_alpha_clip_embeds_fg)
        
        # ip adapter logic
        image_prompt_embeds = self.image_proj_model(projected_alpha_clip_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(self.mlp_projection_layer(torch.zeros_like(clip_image_embeds_fg)))
                
        return image_prompt_embeds, uncond_image_prompt_embeds

    
    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    @torch.inference_mode()
    def generate(
        self,
        pil_image=None,
        alpha=None, 
        prompt=None,
        negative_prompt=None,
        image_prompt_embeds=None,
        uncond_image_prompt_embeds=None,
        scale=1.0,
        num_samples=1,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=50,
        **kwargs,
    ):
        self.set_scale(scale)
        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        if image_prompt_embeds is None or uncond_image_prompt_embeds is None:
            image_prompt_embeds, uncond_image_prompt_embeds= self.get_ipadapter_embeds(pil_image=pil_image, alpha=alpha)
        else:
            image_prompt_embeds = image_prompt_embeds.to(self.device)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(self.device)
            
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.view(bs_embed, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            image=pil_image, 
            mask_image=alpha,
            **kwargs,
        ).images

        return images

