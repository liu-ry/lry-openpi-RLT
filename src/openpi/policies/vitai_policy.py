import dataclasses
from openpi.models import model as _model

import einops
import numpy as np
from openpi import transforms
from PIL import Image  # For resizing


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    if image.shape[:2] != (224, 224):  # Only resize if needed
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((224, 224))  # or Image.BICUBIC
        image = np.asarray(pil_img)
    return image

def _parse_tactile_image(image, bg) -> np.ndarray:
    
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    
    diff = (image.astype(np.float32) - bg.astype(np.float32) + 255.0) / 2.0
    diff = diff.astype(np.uint8)
    image = diff

    if image.shape[:2] != (224, 224):  # Only resize if needed
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((224, 224))  # or Image.BICUBIC
        image = np.asarray(pil_img)
    return image


@dataclasses.dataclass(frozen=True)
class ViTaiInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        
        # Parse base camera images
        # 新数据集 key: cam_top (俯视), cam_wrist (腕部)
        images = {
            "base_0_rgb":       _parse_image(data["images"]["cam_top"]),
            "left_wrist_0_rgb": _parse_image(data["images"]["cam_wrist"]),
        }
        
        image_masks = {key: np.True_ for key in images}
        
        # Add tactile sensors if all are present
        # 新数据集 key: tactile_left, tactile_right
        tactile_keys = ["tactile_left", "tactile_right"]
        if all(key in data["images"] for key in tactile_keys):
            for key in tactile_keys:
                images[key] = _parse_image(data["images"][key])
                image_masks[key] = np.True_
        
        inputs = {"state": state, "image": images, "image_mask": image_masks}
        
        # Actions are only available during training.
        if "actions" in data:
            # Padding from 7 (6 joints + 1 gripper) to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
            
        return inputs



@dataclasses.dataclass(frozen=True)
class ViTaiOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims (6 joints + 1 gripper).
        return {"actions": np.asarray(data["actions"][:, :7])}