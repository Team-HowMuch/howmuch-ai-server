"""Amplitude Agent Analytics bootstrap.

Module-level singletons: `ai` and `agent` must be created once at import
time, not per-request, or session grouping in Agent Analytics breaks.
"""
import os

from amplitude import Amplitude
from amplitude_ai import AIConfig, AmplitudeAI

_api_key = os.environ.get("AMPLITUDE_AI_API_KEY")
if not _api_key:
    print("[amplitude] AMPLITUDE_AI_API_KEY not set — Agent Analytics disabled (dry run)")

ai = AmplitudeAI(
    amplitude=Amplitude(_api_key or "disabled"),
    config=AIConfig(content_mode="full", redact_pii=True, dry_run=not bool(_api_key)),
)

agent = ai.agent(
    "receipt-ocr-vlm",
    description=(
        "Extracts structured receipt data (store, items, total) from a "
        "photographed Korean receipt via VLM + OCR cross-check"
    ),
)
