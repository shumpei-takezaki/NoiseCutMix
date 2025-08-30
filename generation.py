import argparse
from pathlib import Path
import random
from noise_cutmix import NoiseCutMixGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="outputs/", help="output directory")

    parser.add_argument("--device", type=str, default="cuda", help="device to use for generation")
    parser.add_argument("--data_name", type=str, default="cub", help="dataset name", choices=['cub','flower','aircraft'])
    parser.add_argument("--ckpt_path", type=str, default="./ckpts/cub/shot-1-lora-rank10", help="path to the fine-tuned checkpoint")
    
    parser.add_argument("--num_gen", type=int, default=8, help="number of generated images")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="number of inference steps")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha for mask generation")

    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generator = NoiseCutMixGenerator(
        device=args.device, 
        data_name=args.data_name, 
        ckpt_path=args.ckpt_path,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        alpha=args.alpha
    )

    target_labels = [random.randint(0, generator.num_classes - 1) for _ in range(args.num_gen)]

    images, _ = generator(target_labels=target_labels)

    for i, image in enumerate(images):
        image.save(args.output_dir / f"gen_{i:04d}.png")
    print(f"successfuly save generated images to {args.output_dir} !!!")




    

    

