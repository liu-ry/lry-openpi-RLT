import dataclasses

import einops
import numpy as np
from PIL import Image

from openpi import transforms


ROKAE_DUAL_ACTION_DIM = 16


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[:2] != (224, 224):
        image = np.asarray(Image.fromarray(image).resize((224, 224)))
    return image


@dataclasses.dataclass(frozen=True)
class RokaeDualInputs(transforms.DataTransformFn):
    """Inputs for dual Rokae pi0/pi0.5 policies.

    Expected pre-repack data:
    - images: cam_high, cam_left_wrist, cam_right_wrist
    - state: [16]
    - actions: [action_horizon, 16]
    """

    action_dim: int

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != ROKAE_DUAL_ACTION_DIM:
            raise ValueError(f"dual Rokae state must be 16D, got {state.shape}")

        in_images = data["images"]
        required_images = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        missing = [key for key in required_images if key not in in_images]
        if missing:
            raise ValueError(f"dual Rokae images missing {missing}; got {sorted(in_images)}")

        images = {
            "base_0_rgb": _parse_image(in_images["cam_high"]),
            "left_wrist_0_rgb": _parse_image(in_images["cam_left_wrist"]),
            "right_wrist_0_rgb": _parse_image(in_images["cam_right_wrist"]),
        }
        image_masks = {key: np.True_ for key in images}

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": transforms.pad_to_dim(state, self.action_dim),
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != ROKAE_DUAL_ACTION_DIM:
                raise ValueError(f"dual Rokae actions must be 16D, got {actions.shape}")
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RokaeDualOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ROKAE_DUAL_ACTION_DIM])}
