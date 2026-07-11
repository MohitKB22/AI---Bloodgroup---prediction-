"""Domain exceptions. Kept separate from HTTP concerns -- the middleware in
app/middleware/error_handler.py maps each to a status code, so services and
business logic never need to import FastAPI/Starlette themselves.
"""
from __future__ import annotations


class DomainError(Exception):
    """Base class for all expected, handled application errors."""


class ModelNotReadyError(DomainError):
    """Raised when a prediction is requested but no trained model bundle is loaded."""


class InvalidCredentialsError(DomainError):
    pass


class UserAlreadyExistsError(DomainError):
    pass


class InvalidFileTypeError(DomainError):
    pass


class FileTooLargeError(DomainError):
    pass


class ResourceNotFoundError(DomainError):
    pass
