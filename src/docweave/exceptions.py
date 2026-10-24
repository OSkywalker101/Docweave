"""Exception hierarchy for DocWeave.

Every error carries an actionable message. Failures that indicate the *input*
is unusable raise :class:`InputError` subclasses; failures that indicate a
missing optional dependency raise :class:`DependencyMissingError` so the CLI
can tell the user exactly what to install.
"""

from __future__ import annotations


class DocWeaveError(Exception):
    """Base class for all DocWeave errors."""

    exit_code: int = 1


# --------------------------------------------------------------------------
# Input problems
# --------------------------------------------------------------------------


class InputError(DocWeaveError):
    """The supplied input cannot be converted."""

    exit_code = 2


class EmptyFileError(InputError):
    pass


class FileTooLargeError(InputError):
    def __init__(self, size: int, limit: int) -> None:
        super().__init__(
            f"File is {size} bytes which exceeds the configured limit of {limit} bytes. "
            "Raise ConversionConfig.max_file_size_mb if this file should be accepted."
        )
        self.size = size
        self.limit = limit


class UnsupportedFormatError(InputError):
    """The real file format is not supported, or the file is not what it claims."""

    def __init__(self, message: str, detected: str | None = None) -> None:
        super().__init__(message)
        self.detected = detected


class MalformedDocumentError(InputError):
    """The file claims a supported format but its structure is invalid."""


class EncryptedDocumentError(InputError):
    """The document is password protected and cannot be read."""


class LimitExceededError(InputError):
    """A configured processing limit was hit while converting."""

    exit_code = 3


class DeadlineExceededError(LimitExceededError):
    """The configured conversion deadline elapsed."""


# --------------------------------------------------------------------------
# Environment problems
# --------------------------------------------------------------------------


class DependencyMissingError(DocWeaveError):
    """An optional dependency required for the requested operation is absent."""

    exit_code = 4

    def __init__(self, package: str, feature: str, extra: str | None = None) -> None:
        install = extra or package
        super().__init__(
            f"{feature} requires the optional dependency '{package}', which is not installed. "
            f"Install it with: pip install 'docweave[{install}]'  (or: pip install {package})"
        )
        self.package = package
        self.feature = feature
        self.extra = extra


class RenderError(DocWeaveError):
    """The intermediate representation could not be rendered to the target format."""
