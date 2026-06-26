class Error(Exception):
    """Base class for all TensorEval errors."""


class ModelError(Error):
    """Errors while interacting with the model."""


class InvalidModelResponseError(ModelError):
    """Empty or invalid model responses."""


class EmptyModelResponseError(InvalidModelResponseError):
    """Empty model responses."""


class OverlongPromptError(Error):
    """Prompt exceeds model context length."""


class ToolError(Error):
    """Parent class for all tool errors."""


class ToolParseError(ToolError):
    """Errors while parsing tool calls."""


class ToolCallError(ToolError):
    """Errors while calling tools."""


class InfraError(Error):
    """Errors while interacting with infrastructure."""


class SandboxError(InfraError):
    """Errors while interacting with sandboxes."""


class EvaluationError(Error):
    """Errors during evaluation."""


class TrainingError(Error):
    """Errors during training."""


class DeployError(Error):
    """Errors during deployment."""
