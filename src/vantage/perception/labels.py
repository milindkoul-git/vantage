"""Class-label sets.

Label sets belong to the *dataset a model was trained on*, not to the model or
the runtime, so they live here and are referenced by name from the catalog.
A model fine-tuned on a custom dataset registers its own set and everything
else keeps working.
"""

from __future__ import annotations

COCO_80: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)
"""The 80 COCO categories, in the contiguous order detectors output.

Note this is *not* COCO's native 91-id numbering, which has gaps. Detectors
emit dense 0..79 indices, so this ordering is what a decoded ``class_id``
indexes into.
"""

LABEL_SETS: dict[str, tuple[str, ...]] = {
    "coco80": COCO_80,
}


def get_label_set(name: str) -> tuple[str, ...]:
    """Look up a registered label set by name."""
    try:
        return LABEL_SETS[name]
    except KeyError:
        raise KeyError(
            f"unknown label set {name!r}; registered sets are {sorted(LABEL_SETS)}"
        ) from None


def register_label_set(name: str, labels: tuple[str, ...]) -> None:
    """Add a label set, e.g. for a model fine-tuned on custom classes."""
    if not labels:
        raise ValueError("a label set must contain at least one label")
    LABEL_SETS[name] = labels


# Groupings that later phases will care about. Kept as label names rather than
# indices so they survive a change of label set.
PEOPLE: frozenset[str] = frozenset({"person"})
VEHICLES: frozenset[str] = frozenset(
    {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
)
