from __future__ import annotations

from collections import deque
import logging
import os
import random
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


IMAGE_KEY_PRIORITY = (
    "cam_wrist",
    "pikaGripperDepthCamera",
    "pikaGripperFisheyeCamera",
    "cam_front",
    "global_camera",
)


class _TorchUnavailableError(RuntimeError):
    pass


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on robot runtime env.
        raise _TorchUnavailableError("auto critical phase gate requires torch.") from exc
    return torch, nn, F


def _resize_nearest(image: np.ndarray, size: int) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}.")
    h, w = image.shape[:2]
    y_idx = np.linspace(0, h - 1, int(size)).astype(np.int64)
    x_idx = np.linspace(0, w - 1, int(size)).astype(np.int64)
    return image[y_idx][:, x_idx]


def _select_image(observation: dict[str, Any], image_key: str, image_size: int) -> np.ndarray:
    images = observation.get("images")
    if not isinstance(images, dict) or not images:
        raise ValueError("Observation does not contain an images mapping.")
    if image_key == "wrist":
        selected = None
        for key in ("cam_wrist", "pikaGripperDepthCamera", "pikaGripperFisheyeCamera"):
            if key in images:
                selected = images[key]
                break
        if selected is None:
            raise KeyError("No wrist camera image found for auto critical phase gate.")
    elif image_key != "auto":
        if image_key not in images:
            raise KeyError(f"Configured auto critical phase image key {image_key!r} not found.")
        selected = images[image_key]
    else:
        selected = None
        for key in IMAGE_KEY_PRIORITY:
            if key in images:
                selected = images[key]
                break
        if selected is None:
            first_key = sorted(images)[0]
            selected = images[first_key]
    image = np.asarray(selected)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.dtype != np.uint8:
        image = np.asarray(image, dtype=np.float32)
        if float(np.nanmax(image)) <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    if image.shape[0] != image_size or image.shape[1] != image_size:
        image = _resize_nearest(image, image_size)
    return image.astype(np.uint8, copy=False)


class OnlineCriticalPhaseGate:
    """Small online image classifier for critical-phase gating.

    Labels are supplied by the operator's critical on/off decisions. The model is
    intentionally small so it can train alongside actor-critic without owning the
    main learner process.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str,
        image_key: str = "wrist",
        image_size: int = 96,
        min_samples: int = 200,
        train_every_steps: int = 20,
        batch_size: int = 32,
        lr: float = 3e-4,
        enter_threshold: float = 0.75,
        exit_threshold: float = 0.25,
        capacity: int = 4096,
    ) -> None:
        self._torch, self._nn, self._F = _import_torch()
        self._checkpoint_path = checkpoint_path
        self._image_key = image_key
        self._image_size = int(image_size)
        self._min_samples = int(min_samples)
        self._train_every_steps = max(int(train_every_steps), 1)
        self._batch_size = max(int(batch_size), 1)
        self._enter_threshold = float(enter_threshold)
        self._exit_threshold = float(exit_threshold)
        self._buffer: deque[tuple[np.ndarray, int]] = deque(maxlen=int(capacity))
        self._add_count = 0
        self._train_step = 0
        self._lock = threading.Lock()
        self._training = False
        self._device = self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
        self._model = self._make_model().to(self._device)
        self._optimizer = self._torch.optim.AdamW(self._model.parameters(), lr=float(lr), weight_decay=1e-4)
        self._load_checkpoint()

    @property
    def ready(self) -> bool:
        with self._lock:
            if len(self._buffer) < self._min_samples:
                return False
            labels = [label for _, label in self._buffer]
        return any(labels) and not all(labels)

    def observe_label(self, observation: dict[str, Any], label: bool) -> None:
        try:
            image = _select_image(observation, self._image_key, self._image_size)
        except Exception as exc:
            logger.warning("Auto critical phase gate skipped label: %s", exc)
            return
        should_train = False
        with self._lock:
            self._buffer.append((image, int(bool(label))))
            self._add_count += 1
            should_train = self._add_count % self._train_every_steps == 0 and not self._training
            if should_train:
                self._training = True
        if should_train:
            threading.Thread(target=self._train_once_background, daemon=True).start()

    def _train_once_background(self) -> None:
        try:
            self.train_once()
        except Exception as exc:
            logger.warning("Auto critical phase gate background train failed: %s", exc)
        finally:
            with self._lock:
                self._training = False

    def train_once(self) -> dict[str, float] | None:
        with self._lock:
            if len(self._buffer) < self._min_samples:
                return None
            labels_all = [label for _, label in self._buffer]
            if not any(labels_all) or all(labels_all):
                return None
            batch_size = min(self._batch_size, len(self._buffer))
            batch = random.sample(list(self._buffer), batch_size)
        if not batch:
            return None
        images = np.stack([item[0] for item in batch], axis=0)
        labels = np.asarray([item[1] for item in batch], dtype=np.float32)
        x = self._to_tensor(images)
        y = self._torch.from_numpy(labels).to(self._device)
        self._model.train()
        logits = self._model(x).squeeze(-1)
        loss = self._F.binary_cross_entropy_with_logits(logits, y)
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._optimizer.step()
        self._train_step += 1
        if self._train_step <= 3 or self._train_step % 25 == 0:
            self.save_checkpoint()
        prob = self._torch.sigmoid(logits).detach()
        metrics = {
            "loss": float(loss.detach().cpu()),
            "prob_mean": float(prob.mean().cpu()),
            "positive_ratio": float(labels.mean()),
            "buffer_size": float(len(self._buffer)),
            "train_step": float(self._train_step),
        }
        if self._train_step <= 3 or self._train_step % 25 == 0:
            logger.info("Auto critical phase gate train metrics=%s", metrics)
        return metrics

    def predict_probability(self, observation: dict[str, Any]) -> float | None:
        if not self.ready:
            return None
        try:
            image = _select_image(observation, self._image_key, self._image_size)
        except Exception as exc:
            logger.warning("Auto critical phase gate skipped prediction: %s", exc)
            return None
        self._model.eval()
        with self._torch.no_grad():
            logits = self._model(self._to_tensor(image[None])).squeeze()
            return float(self._torch.sigmoid(logits).cpu())

    def desired_phase(self, observation: dict[str, Any], current_phase: bool) -> tuple[bool, float] | None:
        prob = self.predict_probability(observation)
        if prob is None:
            return None
        if current_phase and prob <= self._exit_threshold:
            return False, prob
        if not current_phase and prob >= self._enter_threshold:
            return True, prob
        return current_phase, prob

    def save_checkpoint(self) -> None:
        os.makedirs(os.path.dirname(self._checkpoint_path) or ".", exist_ok=True)
        payload = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "train_step": self._train_step,
            "image_key": self._image_key,
            "image_size": self._image_size,
        }
        tmp_path = f"{self._checkpoint_path}.tmp"
        self._torch.save(payload, tmp_path)
        os.replace(tmp_path, self._checkpoint_path)

    def _load_checkpoint(self) -> None:
        if not os.path.exists(self._checkpoint_path):
            return
        try:
            payload = self._torch.load(self._checkpoint_path, map_location=self._device)
            self._model.load_state_dict(payload["model"])
            if "optimizer" in payload:
                self._optimizer.load_state_dict(payload["optimizer"])
            self._train_step = int(payload.get("train_step", 0))
            logger.info("Loaded auto critical phase gate checkpoint %s", self._checkpoint_path)
        except Exception as exc:
            logger.warning("Failed to load auto critical phase gate checkpoint %s: %s", self._checkpoint_path, exc)

    def _to_tensor(self, images: np.ndarray):
        images = images.astype(np.float32) / 255.0
        images = np.transpose(images, (0, 3, 1, 2))
        return self._torch.from_numpy(images).to(self._device)

    def _make_model(self):
        nn = self._nn

        class ResidualBlock(nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
                self.norm1 = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
                self.norm2 = nn.BatchNorm2d(channels)

            def forward(self, x):
                residual = x
                x = self.norm1(self.conv1(x)).relu()
                x = self.norm2(self.conv2(x))
                return (x + residual).relu()

        class SmallResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 32, 5, stride=2, padding=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                self.blocks = nn.Sequential(
                    ResidualBlock(32),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    ResidualBlock(64),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    ResidualBlock(128),
                )
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 1),
                )

            def forward(self, x):
                return self.head(self.blocks(self.stem(x)))

        return SmallResNet()
