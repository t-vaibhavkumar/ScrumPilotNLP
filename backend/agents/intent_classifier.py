"""
intent_classifier.py — Drop-in replacement for the BART zero-shot classifier.

Uses the fine-tuned DistilBERT model produced by train_colab.ipynb.
Falls back to BART zero-shot automatically if the fine-tuned model
is not found (so the pipeline keeps working before training is done).

Usage in nlp_scrum_extractor.py — replace the _get_classifier() call
with ScrumIntentClassifier, then call .predict(sentences).
"""

from __future__ import annotations

import json
import os
from typing import List, Dict

# Path where the fine-tuned model lives after unzipping
_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "scrum_intent_model"
)

# Module-level singleton
_finetuned_classifier = None
_using_finetuned = False


def _load_finetuned(model_dir: str):
    """Load the fine-tuned DistilBERT classifier."""
    from transformers import pipeline as hf_pipeline
    import torch

    classifier = hf_pipeline(
        "text-classification",
        model=model_dir,
        tokenizer=model_dir,
        device=0 if torch.cuda.is_available() else -1,
    )
    return classifier


def _load_zeroshot_fallback():
    """Load the original BART zero-shot classifier as fallback."""
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,
    )


class ScrumIntentClassifier:
    """
    Unified classifier interface.
    Automatically uses the fine-tuned model if available,
    otherwise falls back to BART zero-shot.
    """

    def __init__(self, model_dir: str = _DEFAULT_MODEL_DIR):
        global _finetuned_classifier, _using_finetuned

        if _finetuned_classifier is None:
            model_dir = os.path.abspath(model_dir)
            if os.path.isdir(model_dir) and os.path.exists(
                os.path.join(model_dir, "config.json")
            ):
                print(f"[Classifier] Loading fine-tuned model from: {model_dir}")
                _finetuned_classifier = _load_finetuned(model_dir)
                _using_finetuned = True
                print("[Classifier] Fine-tuned DistilBERT loaded.")
            else:
                print(
                    f"[Classifier] Fine-tuned model not found at {model_dir}.\n"
                    f"             Falling back to BART zero-shot.\n"
                    f"             Run training/generate_data.py + train_colab.ipynb "
                    f"to get the fine-tuned model."
                )
                _finetuned_classifier = _load_zeroshot_fallback()
                _using_finetuned = False

        self._clf = _finetuned_classifier
        self._is_finetuned = _using_finetuned

    def predict(self, sentences: List[Dict]) -> List[Dict]:
        """
        Classify a list of sentence dicts.

        Args:
            sentences: List of {"speaker": ..., "sentence": ...} dicts.

        Returns:
            Same list enriched with "intent" and "confidence" keys.
        """
        if self._is_finetuned:
            return self._predict_finetuned(sentences)
        else:
            return self._predict_zeroshot(sentences)

    # ── Fine-tuned path ───────────────────────────────────────────────────

    def _predict_finetuned(self, sentences: List[Dict]) -> List[Dict]:
        from backend.config.team_config import CONFIDENCE_THRESHOLD

        texts = [item["sentence"] for item in sentences]

        # Batch prediction
        results = self._clf(texts, batch_size=16)

        enriched = []
        for item, result in zip(sentences, results):
            confidence = round(result["score"], 4)
            enriched.append({
                **item,
                "intent": result["label"],
                "confidence": confidence,
                "low_confidence": confidence < CONFIDENCE_THRESHOLD,
            })

        return enriched

    # ── Zero-shot fallback path ───────────────────────────────────────────

    def _predict_zeroshot(self, sentences: List[Dict]) -> List[Dict]:
        from backend.config.team_config import (
            INTENT_HYPOTHESES, CONFIDENCE_THRESHOLD
        )

        labels = list(INTENT_HYPOTHESES.keys())
        enriched = []

        for item in sentences:
            output = self._clf(
                item["sentence"],
                candidate_labels=labels,
                hypothesis_template="{}",
            )
            top_label = output["labels"][0]
            top_score = output["scores"][0]
            confidence = round(top_score, 4)

            enriched.append({
                **item,
                "intent": top_label,
                "confidence": confidence,
                "low_confidence": confidence < CONFIDENCE_THRESHOLD,
            })

        return enriched

    @property
    def using_finetuned(self) -> bool:
        return self._is_finetuned