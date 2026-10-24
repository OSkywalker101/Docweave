"""DocWeave: a local-first, confidence-guided document conversion engine.

Quick start
-----------

    from docweave import convert

    result = convert("report.pdf")
    print(result.to_markdown())
    for d in result.diagnostics.sorted():
        print(d.to_line())

The Python API never starts a server, never shells out, and never touches the
network. Optional integrations (OCR, HTTP API) live outside this package's
default path.
"""

from __future__ import annotations

__version__ = "0.1.0"

from docweave.config import ConversionConfig
from docweave.converter import (
    JSON,
    JSONL,
    MARKDOWN,
    OUTPUT_FORMATS,
    REPORT,
    BatchItem,
    BatchReport,
    ConversionResult,
    Converter,
    capabilities,
    convert,
    convert_to,
    discover,
    inspect,
)
from docweave.exceptions import (
    DeadlineExceededError,
    DependencyMissingError,
    DocWeaveError,
    EmptyFileError,
    EncryptedDocumentError,
    FileTooLargeError,
    InputError,
    LimitExceededError,
    MalformedDocumentError,
    UnsupportedFormatError,
)
from docweave.models import (
    IR_SCHEMA_VERSION,
    Category,
    Diagnostic,
    DiagnosticReport,
    Document,
    Severity,
)

__all__ = [
    "__version__",
    "IR_SCHEMA_VERSION",
    "ConversionConfig",
    "ConversionResult",
    "Converter",
    "BatchItem",
    "BatchReport",
    "Document",
    "Diagnostic",
    "DiagnosticReport",
    "Category",
    "Severity",
    "convert",
    "convert_to",
    "inspect",
    "capabilities",
    "discover",
    "DocWeaveError",
    "InputError",
    "UnsupportedFormatError",
    "MalformedDocumentError",
    "EncryptedDocumentError",
    "EmptyFileError",
    "FileTooLargeError",
    "LimitExceededError",
    "DeadlineExceededError",
    "DependencyMissingError",
    "MARKDOWN",
    "JSON",
    "JSONL",
    "REPORT",
    "OUTPUT_FORMATS",
]
