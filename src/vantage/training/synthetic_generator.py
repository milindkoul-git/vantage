"""Synthetic multi-scene dataset generator for detection, pose, and HOI interactions."""

from __future__ import annotations

import random

import numpy as np

from vantage.training.dataset import CLASS_TO_ID, AnnotatedSample


class SyntheticDataGenerator:
    """Generates parameterized synthetic training samples with exact ground truth."""

    def __init__(self, width: int = 640, height: int = 480, seed: int = 42) -> None:
        self.width = width
        self.height = height
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def generate_batch(self, count: int = 100) -> list[AnnotatedSample]:
        """Generate a batch of diverse surveillance and HOI training samples."""
        samples: list[AnnotatedSample] = []
        for i in range(count):
            samples.append(self.generate_sample(f"synthetic_{i:05d}.jpg"))
        return samples

    def generate_sample(self, image_id: str) -> AnnotatedSample:
        """Generate one synthetic scene with people and interactive objects."""
        boxes: list[tuple[float, float, float, float, int]] = []
        interactions: list[tuple[int, int, str]] = []

        num_people = self._rng.randint(1, 5)
        for _ in range(num_people):
            # Random person scale and position
            p_w = self._rng.uniform(40, 120)
            p_h = p_w * self._rng.uniform(2.0, 2.8)
            x1 = self._rng.uniform(10, self.width - p_w - 10)
            y1 = self._rng.uniform(10, self.height - p_h - 10)
            p_box = (x1, y1, x1 + p_w, y1 + p_h, CLASS_TO_ID["person"])
            p_idx = len(boxes)
            boxes.append(p_box)

            # 40% chance of holding a phone
            if self._rng.random() < 0.40:
                ph_w, ph_h = p_w * 0.20, p_h * 0.15
                ph_x1 = x1 + p_w * self._rng.uniform(0.1, 0.8)
                ph_y1 = y1 + p_h * self._rng.uniform(0.1, 0.4)
                ph_box = (ph_x1, ph_y1, ph_x1 + ph_w, ph_y1 + ph_h, CLASS_TO_ID["cell_phone"])
                ph_idx = len(boxes)
                boxes.append(ph_box)
                verb = "talking_on_phone" if (ph_y1 - y1) < 0.25 * p_h else "holding_phone"
                interactions.append((p_idx, ph_idx, verb))

            # 30% chance of carrying a backpack
            if self._rng.random() < 0.30:
                bp_w, bp_h = p_w * 0.45, p_h * 0.40
                bp_x1 = x1 + p_w * 0.25
                bp_y1 = y1 + p_h * 0.20
                bp_box = (bp_x1, bp_y1, bp_x1 + bp_w, bp_y1 + bp_h, CLASS_TO_ID["backpack"])
                bp_idx = len(boxes)
                boxes.append(bp_box)
                interactions.append((p_idx, bp_idx, "carrying_baggage"))

            # 25% chance of riding a bicycle
            if self._rng.random() < 0.25:
                bk_w, bk_h = p_w * 1.5, p_h * 0.65
                bk_x1 = max(0, x1 - p_w * 0.25)
                bk_y1 = y1 + p_h * 0.45
                bk_box = (bk_x1, bk_y1, bk_x1 + bk_w, bk_y1 + bk_h, CLASS_TO_ID["bicycle"])
                bk_idx = len(boxes)
                boxes.append(bk_box)
                interactions.append((p_idx, bk_idx, "riding_vehicle"))

        # Add background vehicles
        if self._rng.random() < 0.50:
            c_w = self._rng.uniform(120, 240)
            c_h = c_w * 0.6
            c_x1 = self._rng.uniform(0, self.width - c_w)
            c_y1 = self._rng.uniform(self.height * 0.4, self.height - c_h)
            boxes.append((c_x1, c_y1, c_x1 + c_w, c_y1 + c_h, CLASS_TO_ID["car"]))

        return AnnotatedSample(
            image_path=image_id,
            width=self.width,
            height=self.height,
            boxes=tuple(boxes),
            interactions=tuple(interactions),
        )
