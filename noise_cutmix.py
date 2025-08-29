

from typing import List, Tuple
import random
from PIL import Image

import numpy as np

from utils import make_pipeline,  make_mask_and_ratio

class NoiseCutMixGenerator(object):

    pipe = None  # global sharing is a hack to avoid OOM

    def __init__(
        self,
        data_name = 'cub',
        ckpt_path: str = None,
        device="cuda",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        class_names: List[str] = None,
        meta_class: str = None,
        **kwargs,
    ):

        super(NoiseCutMixGenerator, self).__init__()

        self.pipe = make_pipeline(
            device=device,
            ckpt_path=ckpt_path
        )

        self.class_names = name_to_classes(data_name) if class_names is None else class_names
        self.meta_class = name_to_meta_class(data_name) if class_names is None else meta_class
        self.num_classes = len(self.class_names)

        self.original_labels = [i for i in range(self.num_classes)]

        self.data_name = data_name
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

    def get_aug_mask_and_label(self, target_labels: List[int], resolution: int=512) -> Tuple[List[Image.Image], List[int], List[float]]:
        aug_label = []
        aug_mask = []
        aug_mask_ratio = []

        for _ in range(len(target_labels)):

            label = random.choice(self.original_labels)

            mask, mask_ratio = make_mask_and_ratio(
                size = (resolution,resolution,1),
                alpha=self.aplha,
            )
            
            aug_label.append(label)
            aug_mask.append(mask)
            aug_mask_ratio.append(mask_ratio)

        return aug_mask, aug_label, aug_mask_ratio

    def __call__(
        self,
        target_labels:List[int],
        resolution: int = 512,
        return_mask: bool = False,
        **kwargs,
    ) -> Tuple[List[Image.Image], List[np.ndarray]]:
        
        aug_masks, aug_labels, aug_mask_ratios = self.get_aug_mask_and_label(target_labels, resolution)

        prompts_target = [f"a phot of a {self.pipe.name2placeholder[self.class_names[class_id]]} {self.meta_class}" for class_id in target_labels] # a phot of a <class id> <meta class>
        prompts_aug = [f"a phot of a {self.pipe.name2placeholder[self.class_names[class_id]]} {self.meta_class}" for class_id in aug_labels] # a phot of a <class id> <meta class>

        inputs = dict(
            prompt_1=prompts_target,
            prompt_2=prompts_aug,
            mask=aug_masks,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            num_images_per_prompt=1,
        )

        outputs = self.pipe(**inputs)
        images = outputs.images

        target_labels = [np.eye(self.num_classes)[l] for l in target_labels]
        aug_labels = [np.eye(self.num_classes)[l] for l in aug_labels]

        labels = [mr * sl + (1-mr) * tl for sl, tl, mr in zip(target_labels, aug_labels, aug_mask_ratios)]

        if return_mask:
            return images, labels, aug_masks
        else:
            return images, labels