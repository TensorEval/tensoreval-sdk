"""Error types for the TensorEval SDK.

All exceptions raised by the SDK inherit from :class:`TensorEvalError`,
so callers can catch any SDK-specific failure with a single ``except``.
"""

from __future__ import annotations


class TensorEvalError(Exception):
    """Base exception for all TensorEval SDK errors."""


class APIError(TensorEvalError):
    """Raised when an LLM or MCP provider returns an error response."""


class AuthenticationError(APIError):
    """The API key is missing, invalid, or unauthorized."""


class RateLimitError(APIError):
    """The provider rate-limited the request."""


class TimeoutError(APIError):
    """The request exceeded the configured timeout."""


class MCPError(TensorEvalError):
    """Raised when an MCP server fails during discovery or tool calls."""


class EvaluationError(TensorEvalError):
    """Raised when the evaluation pipeline fails before grading starts."""
