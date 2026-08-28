"""Errors shared by the API and provider adapters."""


class ProviderError(Exception):
    """An error that can be safely represented in an OpenAI-style error body."""

    def __init__(
        self, message: str, *, status_code: int = 502, code: str = "provider_error"
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class ProviderNotConfiguredError(ProviderError):
    """Raised when a requested provider has not been configured."""

    def __init__(self, provider_name: str) -> None:
        super().__init__(
            f"Provider '{provider_name}' is not configured. Set its API key and enable it.",
            status_code=503,
            code="provider_not_configured",
        )


class UnsupportedMessageError(ProviderError):
    """Raised when a provider cannot represent a message in the current API version."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400, code="unsupported_message")
