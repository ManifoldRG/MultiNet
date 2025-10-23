import sys
import os
import math

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

import numpy as np
import pytest

from src.eval_harness.scoring.vqa_metrics import (
    _validate_text_output,
    _normalize_text,
    VQAMetricsCalculator,
)
import src.eval_harness.scoring.vqa_metrics as vqa_metrics_module


def test_validate_text_output():
    assert _validate_text_output("Hello")
    assert _validate_text_output("  world  ")
    assert not _validate_text_output("")
    assert not _validate_text_output("   ")
    assert not _validate_text_output(None)
    assert not _validate_text_output(123)


def test_normalize_text():
    assert _normalize_text("Hello, World!") == "hello world"
    assert _normalize_text("  Multiple    spaces\tand\nlines  ") == "multiple spaces and lines"
    assert _normalize_text("Punctuation: should; be-removed?") == "punctuation should beremoved"
    assert _normalize_text(123) == ""


def test_vqa_metrics_calculator_with_mocked_similarity(monkeypatch):
    class FakeModel:
        def encode(self, text, convert_to_tensor=True):
            return text

    class DummyScore:
        def __init__(self, value):
            self._value = float(value)

        def item(self):
            return self._value

    def fake_cos_sim(a, b):
        return DummyScore(1.0 if a == b else 0.25)

    monkeypatch.setattr(vqa_metrics_module.util, "cos_sim", fake_cos_sim)
    monkeypatch.setattr(vqa_metrics_module, "SentenceTransformer", lambda name: FakeModel())

    calc = VQAMetricsCalculator()

    # Adapter has already normalized the text in extracted_outputs
    predictions = [
        {"raw_output": "A cat on a mat", "extracted_outputs": "a cat on a mat"},  # Adapter normalized
        {"raw_output": "a CAT on a mat!", "extracted_outputs": "a cat on a mat"},  # Adapter normalized
        {"raw_output": "A cat outside", "extracted_outputs": "a cat outside"},  # Adapter normalized
        {"raw_output": "", "extracted_outputs": ""},  # invalid (empty after normalization)
        {"raw_output": "   ", "extracted_outputs": ""},  # invalid (empty after normalization)
    ]

    gts = [
        "A cat on a mat",
        "a CAT on a mat!",
        "A cat on a mat",
        "Does not matter",
        "Also irrelevant",
    ]

    metrics = calc.calculate_metrics(predictions, gts)

    # exact matches: first two are matches after GT normalization -> 2/5
    assert math.isclose(metrics["exact_match_accuracy"], 2 / 5)

    # similarity scores per pair (using fake_cos_sim):
    # Similarity compares normalized pred vs ORIGINAL (unnormalized) GT
    # 1) "a cat on a mat" vs "A cat on a mat" -> not equal (different case) -> 0.25
    # 2) "a cat on a mat" vs "a CAT on a mat!" -> not equal (different case/punct) -> 0.25
    # 3) "a cat outside" vs "A cat on a mat" -> not equal -> 0.25
    # 4) invalid -> 0.0
    # 5) invalid -> 0.0
    expected_sims = [0.25, 0.25, 0.25, 0.0, 0.0]
    assert math.isclose(metrics["avg_similarity_score"], sum(expected_sims) / 5)
    assert math.isclose(metrics["max_similarity_score"], max(expected_sims))
    assert math.isclose(metrics["min_similarity_score"], min(expected_sims))
    assert math.isclose(metrics["similarity_std"], float(np.std(expected_sims)))

    # high similarity threshold is 0.8 in implementation
    assert metrics["high_similarity_threshold"] == 0.8
    high_sim_count = sum(1 for s in expected_sims if s >= 0.8)
    assert math.isclose(metrics["high_similarity_percentage"], high_sim_count / 5 * 100.0)

    # invalid predictions
    assert metrics["total_samples"] == 5
    assert metrics["total_invalid_preds"] == 2
    assert math.isclose(metrics["invalid_percentage"], 2 / 5 * 100.0)

    print(metrics)
    
if __name__ == "__main__":
    pytest.main(['-s', __file__])

    