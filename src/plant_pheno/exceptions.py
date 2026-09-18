"""
Pipeline exception hierarchy.

Provides structured error handling across the application.
"""

from typing import Any


class InatPipelineError(Exception):
    """
    Base exception for all Pipeline errors.

    All custom exceptions inherit from this to allow catching
    all Pipeline-specific errors.
    """

    def __init__(self, message: str, details: dict[Any, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# Persistence layer
class PersistenceError(InatPipelineError):
    """Errors related to persistence layer ."""

    pass


class DBConnectionError(PersistenceError):
    """Errors related to db connection."""

    def __init__(
        self, message: str, file: str | None = None, details: dict | None = None
    ):
        merged = {"file": file} if file else {}
        if details:
            merged.update(details)
        super().__init__(message, details=merged)

    @property
    def file(self) -> str | None:
        return self.details.get("file")


class DBError(PersistenceError):
    """Errors related to db execution."""

    def __init__(
        self, message: str, script: str | None = None, details: dict | None = None
    ):
        merged = {"script": script} if script else {}
        if details:
            merged.update(details)
        super().__init__(message, details=merged)

    @property
    def script(self) -> str | None:
        return self.details.get("script")


# Worflows
class WorkflowError(InatPipelineError):
    """Errors related to workflow execution."""

    pass


## Ingest workflow
class IngestWorkflowError(WorkflowError):
    """Errors related to data ingestion workflow."""

    pass


class SourceReadError(IngestWorkflowError):
    """Errors related to reading source data in ingestion."""

    pass


class ApiEnrichmentError(IngestWorkflowError):
    """Errors related to api enrichment data in ingestion."""

    pass


# Features workflow


## Model training
class TrainModelError(WorkflowError):
    """Errors related to model training workflow."""

    pass


class IncompatiblePipelineModules(TrainModelError):
    """Errors related to incompatible pipeline settings."""

    pass


class TrainPipelineConfigError(TrainModelError):
    """Errors related to PipelineConfig."""

    pass


__all__ = [
    "InatPipelineError",
    "WorkflowError",
    "DBError",
]
