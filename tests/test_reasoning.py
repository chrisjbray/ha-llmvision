"""Thinking controls: request-local overrides, provider mapping, and safe answers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from homeassistant.exceptions import ServiceValidationError

from custom_components.llmvision import ServiceCallData
from custom_components.llmvision.const import (
    DOMAIN,
    ENDPOINT_OPENROUTER,
    REASONING_EFFORTS,
)
from custom_components.llmvision.providers import Google, OpenAI, ProviderFactory


@pytest.fixture(autouse=True)
def no_http_session():
    with patch("custom_components.llmvision.providers.async_get_clientsession"):
        yield


@pytest.fixture
def analysis_call():
    return SimpleNamespace(
        provider="test",
        model="test",
        message="Describe the activity",
        max_tokens=4096,
        base64_images=["frame-0", "frame-1"],
        filenames=["frame-0", "frame-1"],
        response_format="text",
        use_memory=False,
        structure=None,
        model_is_glimpse=lambda: False,
    )


@pytest.mark.parametrize("effort", REASONING_EFFORTS)
def test_service_accepts_effort(effort):
    call = ServiceCallData(SimpleNamespace(data={"reasoning_effort": effort}))
    assert call.reasoning_effort == effort


@pytest.mark.parametrize("effort", ["", "HIGH", "unsupported", 1, False, [], {}])
def test_service_rejects_invalid_effort(effort):
    with pytest.raises(ServiceValidationError, match="reasoning_effort"):
        ServiceCallData(SimpleNamespace(data={"reasoning_effort": effort}))


def test_service_omitted_effort_inherits():
    assert ServiceCallData(SimpleNamespace(data={})).reasoning_effort is None


def test_overrides_do_not_change_provider_or_other_calls(mock_hass, analysis_call):
    settings = {"reasoning_effort": "high", "thinking_budget": 2048}
    mock_hass.data = {DOMAIN: {"test": settings.copy()}}
    provider = OpenAI(
        mock_hass,
        "fake-key",
        "google/gemini-3.5-flash-lite",
        endpoint={"base_url": ENDPOINT_OPENROUTER},
    )
    for effort, expected in [
        (None, "high"),
        ("low", "low"),
        ("none", "none"),
        (None, "high"),
    ]:
        analysis_call.reasoning_effort = effort
        parameters = provider._get_default_parameters(analysis_call)
        assert parameters["reasoning_effort"] == expected
        assert parameters["thinking_budget"] == 2048
        assert mock_hass.data[DOMAIN]["test"] == settings


@pytest.mark.parametrize("builder", ["_prepare_vision_data", "_prepare_text_data"])
@pytest.mark.parametrize(
    "model",
    [
        "google/gemini-3.5-flash-lite",
        "openai/gpt-6-astra",
        "~anthropic/claude-sonnet-latest",
    ],
)
@pytest.mark.parametrize("effort", REASONING_EFFORTS)
def test_openrouter_sends_effort_for_non_allowlisted_models(
    mock_hass, analysis_call, builder, model, effort
):
    analysis_call.reasoning_effort = effort
    provider = OpenAI(
        mock_hass, "fake-key", model, endpoint={"base_url": ENDPOINT_OPENROUTER}
    )
    payload = getattr(provider, builder)(analysis_call)
    assert "reasoning_effort" not in payload
    if effort == "none":
        assert "reasoning" not in payload
    else:
        assert payload["reasoning"] == {"effort": effort}
    assert payload["max_completion_tokens"] == 4096


@pytest.mark.parametrize("builder", ["_prepare_vision_data", "_prepare_text_data"])
def test_direct_openai_keeps_its_native_effort_field(mock_hass, analysis_call, builder):
    analysis_call.reasoning_effort = "high"
    provider = OpenAI(mock_hass, "fake-key", "gpt-5.5")
    payload = getattr(provider, builder)(analysis_call)
    assert payload["reasoning_effort"] == "high"
    assert "reasoning" not in payload


@pytest.mark.parametrize(
    "endpoint", [ENDPOINT_OPENROUTER, {"base_url": ENDPOINT_OPENROUTER}]
)
def test_openrouter_endpoint_forms(mock_hass, endpoint):
    provider = OpenAI(
        mock_hass, "fake-key", "google/gemini-3.5-flash-lite", endpoint=endpoint
    )
    assert provider._model_supports_thinking("medium") == "medium"


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "https://openrouter.ai.example.org/v1",
        "https://example.org/openrouter.ai",
    ],
)
def test_non_openrouter_endpoints_keep_existing_support_check(mock_hass, url):
    provider = OpenAI(
        mock_hass,
        "fake-key",
        "google/gemini-3.5-flash-lite",
        endpoint={"base_url": url},
    )
    assert provider._model_supports_thinking("medium") is False


@pytest.mark.parametrize("builder", ["_prepare_vision_data", "_prepare_text_data"])
@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
def test_google_3_level_wins_over_legacy_budget(
    mock_hass, analysis_call, builder, effort
):
    mock_hass.data = {DOMAIN: {"test": {"thinking_budget": 2048}}}
    analysis_call.reasoning_effort = effort
    provider = Google(mock_hass, "fake-key", "gemini-3.5-flash-lite")
    payload = getattr(provider, builder)(analysis_call)
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingLevel": effort}


@pytest.mark.parametrize("builder", ["_prepare_vision_data", "_prepare_text_data"])
@pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-3.5-flash-lite"])
@pytest.mark.parametrize("budget", [0, 2048])
def test_google_keeps_existing_budget_behavior(
    mock_hass, analysis_call, builder, model, budget
):
    mock_hass.data = {DOMAIN: {"test": {"thinking_budget": budget}}}
    provider = Google(mock_hass, "fake-key", model)
    config = getattr(provider, builder)(analysis_call)["generationConfig"]
    if budget:
        assert config["thinkingConfig"] == {"thinkingBudget": budget}
    else:
        assert "thinkingConfig" not in config


@pytest.mark.parametrize(
    "model,effort", [("gemini-2.5-flash", "medium"), ("gemini-3.5-flash-lite", "xhigh")]
)
def test_google_rejects_unsupported_level_requests(
    mock_hass, analysis_call, model, effort
):
    analysis_call.reasoning_effort = effort
    provider = Google(mock_hass, "fake-key", model)
    with pytest.raises(ServiceValidationError, match="reasoning_effort"):
        provider._prepare_vision_data(analysis_call)


async def test_google_returns_answer_parts_not_thoughts(mock_hass):
    provider = Google(mock_hass, "fake-key", "gemini-3.5-flash-lite")
    provider._post = AsyncMock(
        return_value={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "internal summary"},
                            {"text": "The vehicle "},
                            {"text": "is arriving."},
                        ]
                    }
                }
            ]
        }
    )
    assert await provider._make_request({}) == "The vehicle is arriving."


async def test_google_thought_only_response_is_not_an_answer(mock_hass):
    provider = Google(mock_hass, "fake-key", "gemini-3.5-flash-lite")
    provider._post = AsyncMock(
        return_value={
            "candidates": [
                {"content": {"parts": [{"thought": True, "text": "internal summary"}]}}
            ]
        }
    )
    with pytest.raises(ServiceValidationError, match="No text answer"):
        await provider._make_request({})


async def test_openrouter_does_not_print_authorization(mock_hass, capsys, caplog):
    provider = OpenAI(
        mock_hass,
        "never-log-this-key",
        "google/gemini-3.5-flash-lite",
        endpoint={"base_url": ENDPOINT_OPENROUTER},
    )
    provider._post = AsyncMock(
        return_value={"choices": [{"message": {"content": "ok"}}]}
    )
    assert await provider._make_request({"model": provider.model}) == "ok"
    captured = capsys.readouterr()
    assert "never-log-this-key" not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize(
    "name,model",
    [
        ("OpenRouter", "google/gemini-3.5-flash-lite"),
        ("Google", "gemini-3.5-flash-lite"),
    ],
)
def test_effort_preserves_structured_image_requests(
    mock_hass, analysis_call, name, model
):
    analysis_call.reasoning_effort = "medium"
    analysis_call.response_format = "json"
    analysis_call.structure = {
        "type": "object",
        "properties": {"direction": {"type": "string"}},
        "required": ["direction"],
    }
    provider = ProviderFactory.create(mock_hass, name, {}, model)
    payload = provider._prepare_vision_data(analysis_call)
    if name == "OpenRouter":
        assert payload["response_format"]["json_schema"]["schema"]["required"] == [
            "direction"
        ]
        assert payload["reasoning"] == {"effort": "medium"}
        images = [
            part["image_url"]["url"]
            for part in payload["messages"][1]["content"]
            if part["type"] == "image_url"
        ]
        assert images == [
            "data:image/jpeg;base64,frame-0",
            "data:image/jpeg;base64,frame-1",
        ]
    else:
        assert payload["generationConfig"]["response_mime_type"] == "application/json"
        assert payload["generationConfig"]["thinkingConfig"] == {
            "thinkingLevel": "medium"
        }
        images = [
            part["inline_data"]["data"]
            for part in payload["contents"][1]["parts"]
            if "inline_data" in part
        ]
        assert images == ["frame-0", "frame-1"]


def test_all_analyzer_forms_offer_optional_effort():
    from pathlib import Path

    services = yaml.safe_load(
        (
            Path(__file__).parents[1] / "custom_components/llmvision/services.yaml"
        ).read_text()
    )
    for service in (
        "image_analyzer",
        "video_analyzer",
        "stream_analyzer",
        "data_analyzer",
    ):
        field = services[service]["fields"]["reasoning_effort"]
        assert field["required"] is False
        assert "default" not in field
        assert set(field["selector"]["select"]["options"]) == set(REASONING_EFFORTS)
