"""Unit Tests for Activity Recognition Architecture and Truthful Abstractions."""

from __future__ import annotations

import pytest

from vantage.activity.base import ActivityRecognizer, Recognizer
from vantage.activity.learned import LearnedActionClassifier, LearnedTemporalRecognizer
from vantage.activity.recognizer import (
    HeuristicTemporalRecognizer,
    RuleRecognizer,
)


def test_activity_recognizer_protocol_and_aliases() -> None:
    # 1. Verify ActivityRecognizer protocol alias
    assert ActivityRecognizer is Recognizer

    # 2. Verify HeuristicTemporalRecognizer is primary and RuleRecognizer is alias
    assert RuleRecognizer is HeuristicTemporalRecognizer

    recognizer = HeuristicTemporalRecognizer()
    assert isinstance(recognizer, Recognizer)
    assert isinstance(recognizer, HeuristicTemporalRecognizer)
    assert isinstance(recognizer, RuleRecognizer)

    # 3. Verify LearnedActionClassifier satisfies protocol
    learned = LearnedActionClassifier()
    assert isinstance(learned, Recognizer)

    # 4. Verify LearnedTemporalRecognizer raises NotImplementedError on instantiate
    with pytest.raises(NotImplementedError):
        LearnedTemporalRecognizer()
