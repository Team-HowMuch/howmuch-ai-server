"""Amplitude Agent Analytics 배선 검증 (mock — 실제 네트워크 호출 없음)."""
from amplitude_ai.core.tracking import (
    PROP_COST_USD,
    PROP_INPUT_TOKENS,
    PROP_LATENCY_MS,
    PROP_MODEL_NAME,
    PROP_OUTPUT_TOKENS,
    PROP_PROVIDER,
    PROP_SESSION_ID,
)
from amplitude_ai.testing import MockAmplitudeAI


def test_receipt_ocr_agent_emits_events():
    mock = MockAmplitudeAI()
    agent = mock.agent("receipt-ocr-vlm")
    with agent.session(device_id="req-1", session_id="req-1") as s:
        s.track_user_message("Extract structured data from a photographed Korean receipt")
        s.track_ai_message(
            "{\"store_name\": \"Starfield\"}",
            "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            "openai-compatible-proxy",
            1200,
            input_tokens=512,
            output_tokens=64,
            total_tokens=576,
        )
    assert len(mock.events_for_agent("receipt-ocr-vlm")) > 0
    mock.assert_event_tracked("[Agent] User Message")
    mock.assert_session_closed("req-1")


def test_data_quality_gate():
    """Agent Analytics 대시보드가 깨지지 않도록 필수 필드가 채워졌는지 확인."""
    mock = MockAmplitudeAI()
    agent = mock.agent("receipt-ocr-vlm")
    with agent.session(device_id="req-1", session_id="req-1") as s:
        s.track_user_message("Extract structured data from a photographed Korean receipt")
        s.track_ai_message(
            "{\"store_name\": \"Starfield\"}",
            "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            "openai-compatible-proxy",
            1200,
            input_tokens=512,
            output_tokens=64,
            total_tokens=576,
        )

    ai_events = mock.get_events("[Agent] AI Response")
    for e in ai_events:
        p = e.event_properties or {}
        assert e.user_id or e.device_id
        assert p.get(PROP_SESSION_ID)
        assert p.get(PROP_MODEL_NAME)
        assert p.get(PROP_PROVIDER)
        assert p.get(PROP_LATENCY_MS, 0) > 0
        assert p.get(PROP_INPUT_TOKENS, 0) > 0
        assert p.get(PROP_OUTPUT_TOKENS, 0) > 0
        # 자체 호스팅 Qwen 모델은 genai-prices에 없어 비용이 비어 있을 수 있다 — 정상.


def test_local_backend_sets_zero_cost():
    """mlx-local 백엔드는 과금이 없으므로 total_cost_usd=0을 명시해야 한다."""
    mock = MockAmplitudeAI()
    agent = mock.agent("receipt-ocr-vlm")
    with agent.session(device_id="req-2", session_id="req-2") as s:
        s.track_user_message("Extract structured data from a photographed Korean receipt")
        s.track_ai_message(
            "{\"store_name\": \"Starfield\"}",
            "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "mlx-local",
            800,
            total_cost_usd=0,
        )
    ai_events = mock.get_events("[Agent] AI Response")
    p = ai_events[-1].event_properties or {}
    assert p.get(PROP_COST_USD) == 0
