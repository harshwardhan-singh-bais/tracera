"""Tests for OpenAI-provider error classification (_classify_error).

Permanent failures (401/402/403/404) must map to ProviderUnavailableError so
failover skips the provider; rate limits (429) stay transient; 500s and
connection errors are generic ProviderError.
"""

import httpx
import openai
import pytest

from tracera.errors import (
    ProviderAuthError,
    ProviderContextLengthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from tracera.providers.openai_provider import _classify_error


def _api_error(cls, status: int, message: str):
    request = httpx.Request("POST", "https://api.test/v1")
    response = httpx.Response(status, request=request)
    return cls(message, response=response, body=None)


def test_classify_401_auth_is_permanent():
    err = _api_error(openai.AuthenticationError, 401, "invalid api key")
    assert isinstance(_classify_error(err), ProviderAuthError)


def test_classify_402_payment_required_is_permanent():
    # 402 is not a dedicated SDK class — it surfaces as the base APIStatusError.
    class _PaymentRequired(openai.APIStatusError):
        pass

    err = _api_error(_PaymentRequired, 402, "PAYMENT_METHOD_REQUIRED")
    assert isinstance(_classify_error(err), ProviderUnavailableError)


def test_classify_403_forbidden_is_permanent():
    err = _api_error(openai.PermissionDeniedError, 403, "no access")
    assert isinstance(_classify_error(err), ProviderUnavailableError)


def test_classify_404_model_not_found_is_permanent():
    err = _api_error(openai.NotFoundError, 404, "model does not exist")
    assert isinstance(_classify_error(err), ProviderUnavailableError)


def test_classify_429_rate_limit_is_transient():
    err = _api_error(openai.RateLimitError, 429, "rate limited")
    assert isinstance(_classify_error(err), ProviderRateLimitError)


def test_classify_400_context_too_long():
    err = _api_error(openai.BadRequestError, 400, "maximum context length exceeded")
    assert isinstance(_classify_error(err), ProviderContextLengthError)


def test_classify_500_is_generic():
    err = _api_error(openai.InternalServerError, 500, "boom")
    assert isinstance(_classify_error(err), ProviderError)
    assert not isinstance(_classify_error(err), ProviderUnavailableError)


def test_classify_connection_error_is_generic():
    request = httpx.Request("POST", "https://api.test/v1")
    err = openai.APIConnectionError(request=request)
    assert isinstance(_classify_error(err), ProviderError)


def test_classify_unknown_exception_is_generic():
    assert isinstance(_classify_error(RuntimeError("weird")), ProviderError)
