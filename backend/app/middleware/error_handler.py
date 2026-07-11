"""Centralized exception -> HTTP response mapping, registered once in
main.py. Keeps endpoint code free of repetitive try/except-to-HTTPException
boilerplate for anything that isn't already an intentional HTTPException.
"""
from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    DomainError,
    FileTooLargeError,
    InvalidCredentialsError,
    InvalidFileTypeError,
    ModelNotReadyError,
    ResourceNotFoundError,
    UserAlreadyExistsError,
)

logger = logging.getLogger(__name__)

_STATUS_BY_EXCEPTION: dict[type[DomainError], int] = {
    ModelNotReadyError: status.HTTP_503_SERVICE_UNAVAILABLE,
    InvalidCredentialsError: status.HTTP_401_UNAUTHORIZED,
    UserAlreadyExistsError: status.HTTP_409_CONFLICT,
    InvalidFileTypeError: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    FileTooLargeError: status.HTTP_413_CONTENT_TOO_LARGE,
    ResourceNotFoundError: status.HTTP_404_NOT_FOUND,
}


def register_exception_handlers(app: FastAPI) -> None:
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(DomainError)
    async def handle_domain_error(request: Request, exc: DomainError) -> JSONResponse:
        status_code = _STATUS_BY_EXCEPTION.get(type(exc), status.HTTP_400_BAD_REQUEST)
        return JSONResponse(status_code=status_code, content={"detail": str(exc), "request_id": getattr(request.state, "request_id", None)})

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "request_id": getattr(request.state, "request_id", None)},
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled exception [request_id=%s]", request_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred.", "request_id": request_id},
        )
