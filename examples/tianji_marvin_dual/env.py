"""OpenPI runtime adapter for dual TianjiMarvin AR arms with Zhixing grippers."""
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import einops
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.tianji_marvin_dual import constants
from examples.tianji_marvin_dual import real_env as _real_env


class DualTianjiMarvinEnvironment(_environment.Environment):
    """Runtime observation/action adapter.

    Observation:
        {"state": float32[16], "images": {name: uint8[3,H,W]}}

    Action:
        {"actions": float[16]}
    """

    def __init__(
        self,
        *,
        reset_joint_positions: Optional[list[float]] = None,
        render_height: int = 224,
        render_width: int = 224,
        obs_ready_timeout_s: float | None = 10.0,
        enable_tactile: bool = False,
        tactile_image_provider: Callable[[], dict[str, np.ndarray]] | None = None,
        **real_env_kwargs,
    ) -> None:
        self._env = _real_env.make_real_env(
            reset_joint_positions=reset_joint_positions,
            image_resize_hw=(render_height, render_width),
            obs_ready_timeout_s=obs_ready_timeout_s,
            enable_tactile=enable_tactile,
            tactile_image_provider=tactile_image_provider,
            **real_env_kwargs,
        )
        self._render_height = render_height
        self._render_width = render_width
        self._ts = None

    @override
    def reset(self) -> None:
        if self._ts is None:
            self._env.connect()
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("reset() must be called before get_observation()")
        obs = self._ts.observation
        images_out = {}
        for cam_name, img in obs["images"].items():
            img_resized = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, self._render_height, self._render_width)
            )
            images_out[cam_name] = einops.rearrange(img_resized, "h w c -> c h w")
        return {
            "state": obs["qpos"].astype(np.float32),
            "images": images_out,
        }

    @override
    def apply_action(self, action: dict) -> None:
        try:
            self._ts = self._env.step(action["actions"])
        except Exception:
            self._env.arms.stop()
            raise

    def close(self) -> None:
        self._env.disconnect()
