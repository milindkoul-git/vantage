"""Model-family adapters: input shaping and output decoding.

An adapter knows one thing — the input/output convention of a family of models
— and nothing about how that model is executed. Adding RT-DETR or D-FINE later
means adding a file here, not touching engines, backends or contracts.
"""

from vantage.perception.adapters.base import ModelAdapter, PreparedInput
from vantage.perception.adapters.dfine import DFineAdapter
from vantage.perception.adapters.grounding_dino import GroundingDinoAdapter
from vantage.perception.adapters.yolox import YoloxAdapter

ADAPTERS: dict[str, type[ModelAdapter]] = {
    "yolox": YoloxAdapter,
    "dfine": DFineAdapter,
    "grounding-dino": GroundingDinoAdapter,
}


def get_adapter(name: str) -> type[ModelAdapter]:
    try:
        return ADAPTERS[name]
    except KeyError:
        raise KeyError(
            f"unknown model adapter {name!r}; registered adapters are {sorted(ADAPTERS)}"
        ) from None


__all__ = [
    "ADAPTERS",
    "DFineAdapter",
    "GroundingDinoAdapter",
    "ModelAdapter",
    "PreparedInput",
    "YoloxAdapter",
    "get_adapter",
]
