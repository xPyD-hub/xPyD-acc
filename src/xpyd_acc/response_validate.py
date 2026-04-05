"""Endpoint response validation for OpenAI Chat Completions API schema.

Validates that API responses contain the expected structure before processing,
catching malformed responses early with clear error messages.
"""

from __future__ import annotations

from typing import Any


class ResponseValidationError(Exception):
    """Raised when an API response does not conform to expected schema."""


def validate_chat_response(data: Any, *, require_logprobs: bool = False) -> None:
    """Validate a Chat Completions API response.

    Args:
        data: Parsed JSON response from the API.
        require_logprobs: If True, also validate that logprobs data is present.

    Raises:
        ResponseValidationError: If the response is malformed.
    """
    if not isinstance(data, dict):
        raise ResponseValidationError(
            f"Response must be a JSON object, got {type(data).__name__}"
        )

    if "choices" not in data:
        raise ResponseValidationError(
            "Response missing 'choices' field. "
            "Is this an OpenAI-compatible endpoint?"
        )

    choices = data["choices"]
    if not isinstance(choices, list):
        raise ResponseValidationError(
            f"'choices' must be an array, got {type(choices).__name__}"
        )

    if len(choices) == 0:
        raise ResponseValidationError("'choices' array is empty — no completion returned")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise ResponseValidationError(
            f"choices[0] must be an object, got {type(choice).__name__}"
        )

    if "message" not in choice:
        raise ResponseValidationError(
            "choices[0] missing 'message' field"
        )

    message = choice["message"]
    if not isinstance(message, dict):
        raise ResponseValidationError(
            f"choices[0].message must be an object, got {type(message).__name__}"
        )

    if "content" not in message:
        raise ResponseValidationError(
            "choices[0].message missing 'content' field"
        )

    if require_logprobs:
        logprobs = choice.get("logprobs")
        if logprobs is None:
            raise ResponseValidationError(
                "choices[0] missing 'logprobs' — did you include logprobs=true in the request?"
            )
        if not isinstance(logprobs, dict):
            raise ResponseValidationError(
                f"choices[0].logprobs must be an object, got {type(logprobs).__name__}"
            )
        content = logprobs.get("content")
        if content is None:
            raise ResponseValidationError(
                "choices[0].logprobs missing 'content' array"
            )
        if not isinstance(content, list):
            raise ResponseValidationError(
                f"choices[0].logprobs.content must be an array, got {type(content).__name__}"
            )
