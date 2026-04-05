"""Tests for response_validate module."""

from __future__ import annotations

import pytest

from xpyd_acc.response_validate import ResponseValidationError, validate_chat_response

# ---------------------------------------------------------------------------
# Valid responses
# ---------------------------------------------------------------------------


def _valid_response(*, with_logprobs: bool = True) -> dict:
    """Return a valid OpenAI Chat Completions response."""
    choice: dict = {
        "index": 0,
        "message": {"role": "assistant", "content": "Hello!"},
        "finish_reason": "stop",
    }
    if with_logprobs:
        choice["logprobs"] = {
            "content": [
                {
                    "token": "Hello",
                    "logprob": -0.1,
                    "top_logprobs": [{"token": "Hello", "logprob": -0.1}],
                },
                {
                    "token": "!",
                    "logprob": -0.05,
                    "top_logprobs": [{"token": "!", "logprob": -0.05}],
                },
            ]
        }
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [choice],
    }


def test_valid_response_passes():
    """A well-formed response should pass without error."""
    validate_chat_response(_valid_response())


def test_valid_response_with_logprobs():
    """Validation with require_logprobs=True on a response that has logprobs."""
    validate_chat_response(_valid_response(with_logprobs=True), require_logprobs=True)


def test_valid_response_without_logprobs_not_required():
    """No logprobs and require_logprobs=False should pass."""
    validate_chat_response(_valid_response(with_logprobs=False), require_logprobs=False)


# ---------------------------------------------------------------------------
# Invalid: top-level issues
# ---------------------------------------------------------------------------


def test_not_a_dict():
    with pytest.raises(ResponseValidationError, match="JSON object"):
        validate_chat_response("not a dict")


def test_not_a_dict_list():
    with pytest.raises(ResponseValidationError, match="JSON object"):
        validate_chat_response([1, 2, 3])


def test_missing_choices():
    with pytest.raises(ResponseValidationError, match="missing 'choices'"):
        validate_chat_response({"id": "x", "model": "m"})


def test_choices_not_a_list():
    with pytest.raises(ResponseValidationError, match="must be an array"):
        validate_chat_response({"choices": "bad"})


def test_choices_empty():
    with pytest.raises(ResponseValidationError, match="empty"):
        validate_chat_response({"choices": []})


# ---------------------------------------------------------------------------
# Invalid: choice-level issues
# ---------------------------------------------------------------------------


def test_choice_not_a_dict():
    with pytest.raises(ResponseValidationError, match="choices\\[0\\] must be an object"):
        validate_chat_response({"choices": ["bad"]})


def test_choice_missing_message():
    with pytest.raises(ResponseValidationError, match="missing 'message'"):
        validate_chat_response({"choices": [{"index": 0}]})


def test_message_not_a_dict():
    with pytest.raises(ResponseValidationError, match="message must be an object"):
        validate_chat_response({"choices": [{"message": "bad"}]})


def test_message_missing_content():
    with pytest.raises(ResponseValidationError, match="missing 'content'"):
        validate_chat_response({"choices": [{"message": {"role": "assistant"}}]})


# ---------------------------------------------------------------------------
# Invalid: logprobs issues (require_logprobs=True)
# ---------------------------------------------------------------------------


def test_logprobs_missing_when_required():
    resp = _valid_response(with_logprobs=False)
    with pytest.raises(ResponseValidationError, match="missing 'logprobs'"):
        validate_chat_response(resp, require_logprobs=True)


def test_logprobs_not_a_dict():
    resp = _valid_response(with_logprobs=False)
    resp["choices"][0]["logprobs"] = "bad"
    with pytest.raises(ResponseValidationError, match="logprobs must be an object"):
        validate_chat_response(resp, require_logprobs=True)


def test_logprobs_missing_content():
    resp = _valid_response(with_logprobs=False)
    resp["choices"][0]["logprobs"] = {}
    with pytest.raises(ResponseValidationError, match="missing 'content' array"):
        validate_chat_response(resp, require_logprobs=True)


def test_logprobs_content_not_a_list():
    resp = _valid_response(with_logprobs=False)
    resp["choices"][0]["logprobs"] = {"content": "bad"}
    with pytest.raises(ResponseValidationError, match="must be an array"):
        validate_chat_response(resp, require_logprobs=True)
