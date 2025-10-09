from typing import List, Optional
from PIL import Image
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class CustomGenPipeline:

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        multimodal_encoder: AutoModelForCausalLM,
    ):
        super().__init__()
        self.multimodal_encoder = multimodal_encoder
        self.tokenizer = tokenizer

    @torch.no_grad()
    def __call__(
        self,
        inputs: List[Image.Image | str] | str | Image.Image,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.5,
        crop_info: List[int] = [0, 0],
        original_size: List[int] = [1024, 1024],
        generator = None,
    ):
        if not isinstance(inputs, list):
            inputs = [inputs]

        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        np_img = self._prepare_and_encode_inputs(
            inputs,
            do_classifier_free_guidance,
            generator=generator,
            guidance_scale=guidance_scale
        )
        images = self.numpy_to_pil(np_img)

        return images[0]

    def _prepare_and_encode_inputs(
        self,
        inputs: List[str | Image.Image],
        do_classifier_free_guidance: bool = False,
        generator=None,
        guidance_scale: float = 4.5,
    ):
        prompt = self.multimodal_encoder.generate_image(
            text=inputs, tokenizer=self.tokenizer, generator=generator
        )
        return prompt

    def numpy_to_pil(self, images: np.ndarray) -> List[Image.Image]:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # Special case for grayscale (single channel) images.
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        return pil_images
