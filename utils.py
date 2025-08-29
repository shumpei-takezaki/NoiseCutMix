import os
from typing import Tuple
from PIL import Image

import numpy as np
from tqdm.auto import tqdm

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DPMSolverMultistepScheduler

from pipeline import NoiseCutMixPipeline

ERROR_MESSAGE = "Tokenizer already contains the token {token}. \
Please pass a different `token` that is not already in the tokenizer."

def make_mask_and_ratio(size: Tuple[int, ...], alpha: float) -> Image.Image:
    """
    Generate a binary mask image with a random rectangular region set to 1 (white) 
    and the rest set to 0 (black), based on a lambda value sampled from the Beta distribution.

    Args:
        size (Tuple[int, ...]): Input image shape (H, W, C).
        alpha (float): Hyperparameter for the Beta distribution used to sample lambda.

    Returns:
        PIL.Image.Image: A grayscale mask image with values 0 (background) and 255 (rectangular region).
    """
    if len(size) == 4:
        H, W = size[1], size[2]
    elif len(size) == 3:
        H, W = size[0], size[1]
    else:
        raise ValueError("Input size must be (H, W, C) or (B, H, W, C)")
    
    lam = np.random.beta(alpha, alpha)

    cut_ratio = np.sqrt(1. - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[bby1:bby2, bbx1:bbx2] = 255

    masked_area = (bbx2 - bbx1) * (bby2 - bby1)
    total_area = W * H
    area_ratio = masked_area / total_area

    return Image.fromarray(mask), area_ratio

def load_diffmix_embeddings(
    embed_path: str,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
):

    embedding_ckpt = torch.load(embed_path, map_location="cpu")
    learned_embeds_dict = embedding_ckpt["learned_embeds_dict"]
    name2placeholder = embedding_ckpt["name2placeholder"]
    placeholder2name = embedding_ckpt["placeholder2name"]

    name2placeholder = {
        k.replace("/", " ").replace("_", " "): v for k, v in name2placeholder.items()
    }
    placeholder2name = {
        v: k.replace("/", " ").replace("_", " ") for k, v in name2placeholder.items()
    }

    for token, token_embedding in tqdm(learned_embeds_dict.items()):

        # add the token in tokenizer
        num_added_tokens = tokenizer.add_tokens(token)
        assert num_added_tokens > 0, ERROR_MESSAGE.format(token=token)

        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))
        added_token_id = tokenizer.convert_tokens_to_ids(token)

        # get the old word embeddings
        embeddings = text_encoder.get_input_embeddings()

        # get the id for the token and assign new embeds
        embeddings.weight.data[added_token_id] = token_embedding.to(
            embeddings.weight.dtype
        )

    return name2placeholder, placeholder2name

def parse_finetuned_ckpt(finetuned_ckpt):
    lora_path = None
    embed_path = None
    for file in os.listdir(finetuned_ckpt):
        if "pytorch_lora_weights" in file:
            lora_path = os.path.join(finetuned_ckpt, file)
        elif "learned_embeds-steps-last" in file:
            embed_path = os.path.join(finetuned_ckpt, file)
    return lora_path, embed_path

def make_pipeline(device, ckpt_path=None):

    pipe = NoiseCutMixPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5",
        use_auth_token=True,
        revision=None,
        local_files_only=False,
        torch_dtype=torch.float16
    ).to(device)

    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, local_files_only=False)

    if ckpt_path is not None:
        lora_path, embed_path = parse_finetuned_ckpt(ckpt_path)
        name2placeholder, placeholder2name = load_diffmix_embeddings(
                embed_path,
                pipe.text_encoder,
                pipe.tokenizer,
            )
        print(f"successfuly load lora weights from {lora_path}! ! ! ")
        pipe.load_lora_weights(lora_path)
        pipe.scheduler = scheduler
        pipe.safety_checker = None
        pipe.name2placeholder = name2placeholder
        pipe.placeholder2name = placeholder2name
    else:
        pipe.scheduler = scheduler
    
    return pipe