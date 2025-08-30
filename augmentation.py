import argparse
from pathlib import Path
import random

import numpy as np

from utils import num_to_groups
from noise_cutmix import NoiseCutMixGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="outputs/", help="output directory")

    parser.add_argument("--device", type=str, default="cuda", help="device to use for generation")
    parser.add_argument("--data_name", type=str, default="cub", help="dataset name", choices=['cub','flower','aircraft'])
    parser.add_argument("--ckpt_path", type=str, default="./ckpts/cub/shot-1-lora-rank10", help="path to the fine-tuned checkpoint")
    
    parser.add_argument("--batch_size", type=int, default=8, help="number of images to generate in a batch")
    parser.add_argument("--num_aug", type=int, default=10, help="number of augmented images")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="number of inference steps")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha for mask generation")

    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "image").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mask").mkdir(parents=True, exist_ok=True) 

    generator = NoiseCutMixGenerator(
        device=args.device, 
        data_name=args.data_name, 
        ckpt_path=args.ckpt_path,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        alpha=args.alpha
    )

    split_batches = num_to_groups(args.num_aug, args.batch_size)

    save_labels = []
    for i, batch in enumerate(split_batches):
        target_labels = [random.randint(0, generator.num_classes - 1) for _ in range(batch)]
        images, labels, masks = generator(target_labels=target_labels, return_mask=True)

        for j, (image, mask) in enumerate(zip(images, masks)):
            image.save(args.output_dir / "image" / f"gen_{i * args.batch_size + j:04d}.png")
            mask.save(args.output_dir / "mask" / f"mask_{i * args.batch_size + j:04d}.png")
        
        save_labels.extend(labels)

        print(f"finished {i+1}/{len(split_batches)}")
    
    save_labels = np.array(save_labels)
    np.save(args.output_dir / "soft_label.npy", save_labels)
    
    print(f"successfuly save augmented images to {args.output_dir} !!!")