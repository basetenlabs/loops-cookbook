"""The run's one Chat Completions listener: authentication, routing, and delivery.

Every endpoint requires an attempt credential, which selects the attempt's
``HttpModelAccess``; nothing in a request body or header can change the attempt or
the producer a credential names. Beside the OpenAI endpoints, ``POST
/v1/loops/discard`` takes back a completion the agent declined, so it does not
train. Requests rejected before sampling create no evidence. A response is settled
as delivered only after the server has written all of it, a stream's termination
included.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, cast

from aiohttp import web
from pydantic import ValidationError

from loops_cookbook.evidence import InternalPolicyDeltaError, StaleRolloutError
from loops_cookbook.llm.client import (
    ContextWindowExceeded,
    ConversationBusy,
    UnsupportedParameterError,
)
from loops_cookbook.llm.rendering import RenderingError
from loops_cookbook.model_access.api import ModelAccessError
from loops_cookbook.model_access.openai.access import (
    AccessEnded,
    HttpModelAccess,
    IdempotencyConflict,
    UnknownGeneration,
    request_digest,
)
from loops_cookbook.model_access.openai.policy import (
    InvalidRequestError,
    ModelNotFoundError,
    admit,
)
from loops_cookbook.model_access.openai.schema.models import CreateChatCompletionRequest
from loops_cookbook.model_access.openai.translate import Completion, InvalidOutput, error_body

logger = logging.getLogger(__name__)

CONVERSATION_HEADER = "X-Loops-Conversation-Id"
IDEMPOTENCY_HEADER = "Idempotency-Key"
DISCARD_PATH = "/v1/loops/discard"
# Base64 images make requests large; this bounds one request, not a conversation.
MAX_REQUEST_BYTES = 64 * 1024 * 1024
_MAX_HEADER_VALUE = 256


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    """``[runtime.openai]``: where the gateway listens, and what it advertises."""

    bind: str
    advertised_base_url: str | None
    model: str
    # How long closing the listener waits for handlers of attempts still running.
    shutdown_seconds: float

    def host_port(self) -> tuple[str, int]:
        """``bind`` split; ``[runtime.openai]`` has already checked it is host:port."""
        host, _, port = self.bind.rpartition(":")
        return host.strip("[]"), int(port)


class _HttpError(Exception):
    def __init__(
        self, status: int, message: str, *, kind: str, code: str | None, param: str | None = None
    ) -> None:
        super().__init__(message)
        self.status, self.kind, self.code, self.param = status, kind, code, param

    def response(self) -> web.Response:
        body = error_body(str(self), kind=self.kind, code=self.code, param=self.param)
        return web.json_response(body, status=self.status)


def _invalid(message: str, *, code: str | None = None, param: str | None = None) -> _HttpError:
    return _HttpError(400, message, kind="invalid_request_error", code=code, param=param)


class Gateway:
    """The listener attempts register credentials on; started at its first connection."""

    def __init__(self, settings: GatewaySettings) -> None:
        self._settings = settings
        self._credentials: dict[str, tuple[HttpModelAccess, str]] = {}
        self._runner: web.AppRunner | None = None
        self._base_url: str | None = None
        self._starting = asyncio.Lock()
        self._closed = False

    @property
    def model(self) -> str:
        return self._settings.model

    async def base_url(self) -> str:
        """The advertised origin plus ``/v1``, binding the listener on first use."""
        async with self._starting:
            if self._closed:
                raise RuntimeError("the gateway is closed")
            if self._base_url is None:
                self._base_url = await self._start()
        return self._base_url

    async def _start(self) -> str:
        host, port = self._settings.host_port()
        app = web.Application(client_max_size=MAX_REQUEST_BYTES)
        app.router.add_route("*", "/{path:.*}", self._handle)
        runner = web.AppRunner(
            app,
            access_log=None,
            handler_cancellation=True,
            shutdown_timeout=self._settings.shutdown_seconds,
        )
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        try:
            await site.start()
        except BaseException:
            await runner.cleanup()
            raise
        self._runner = runner
        bound = runner.addresses[0][1]
        origin = self._settings.advertised_base_url or f"http://127.0.0.1:{bound}"
        return origin.rstrip("/") + "/v1"

    def register(self, credential: str, access: HttpModelAccess, producer: str) -> None:
        self._credentials[credential] = (access, producer)

    def revoke(self, credential: str) -> None:
        self._credentials.pop(credential, None)

    async def close(self) -> None:
        """Stop listening and revoke every credential still registered."""
        self._closed = True
        self._credentials.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    # ------------------------------------------------------------- handlers

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        try:
            access, producer = self._authenticate(request)
            path, method = request.path, request.method
            if path == "/v1/chat/completions" and method == "POST":
                return await self._completions(request, access, producer)
            if path == DISCARD_PATH and method == "POST":
                return await self._discard(request, access, producer)
            if path == "/v1/models" and method == "GET":
                return web.json_response({"object": "list", "data": [self._model_card()]})
            if path.startswith("/v1/models/") and method == "GET":
                if path.removeprefix("/v1/models/") != self.model:
                    raise _HttpError(
                        404,
                        "The model does not exist",
                        kind="invalid_request_error",
                        code="model_not_found",
                        param="model",
                    )
                return web.json_response(self._model_card())
            raise _HttpError(
                404,
                f"Unknown request URL: {method} {path}",
                kind="invalid_request_error",
                code="unknown_url",
            )
        except _HttpError as error:
            return error.response()
        except AccessEnded as error:
            return _HttpError(
                401, str(error), kind="invalid_request_error", code="invalid_api_key"
            ).response()
        except Exception:  # a server fault still gets a canonical body
            logger.exception("the Chat Completions gateway failed a request")
            return _server_error().response()

    def _authenticate(self, request: web.Request) -> tuple[HttpModelAccess, str]:
        headers = request.headers.getall("Authorization", [])
        token = headers[0].removeprefix("Bearer ") if len(headers) == 1 else ""
        found = (
            self._credentials.get(token) if headers and headers[0].startswith("Bearer ") else None
        )
        if found is None:
            raise _HttpError(
                401,
                "Invalid or revoked API key",
                kind="invalid_request_error",
                code="invalid_api_key",
            )
        return found

    def _model_card(self) -> dict[str, object]:
        return {"id": self.model, "object": "model", "created": 0, "owned_by": "loops"}

    async def _completions(
        self, request: web.Request, access: HttpModelAccess, producer: str
    ) -> web.StreamResponse:
        handler = asyncio.current_task()
        if handler is not None:
            access.own(handler)  # sealing waits for, or cancels, what the attempt started
        conversation = _header(request, CONVERSATION_HEADER)
        key = _header(request, IDEMPOTENCY_HEADER)
        body = await _json(request)
        try:
            parsed = CreateChatCompletionRequest.model_validate(body)
        except ValidationError as error:
            first = error.errors()[0]
            param = _path(first["loc"])
            raise _invalid(first["msg"], code="invalid_value", param=param) from None
        try:
            chat = admit(parsed, model=self.model)
            task = access.complete(
                producer,
                chat,
                sent=cast(dict[str, Any], body),
                conversation_id=conversation,
                key=key,
                digest=request_digest(parsed, conversation),
            )
        except UnsupportedParameterError as error:
            raise _invalid(str(error), code="unsupported_parameter", param=error.param) from None
        except InvalidRequestError as error:
            raise _invalid(str(error), code="invalid_value", param=error.param) from None
        except ModelNotFoundError as error:
            raise _HttpError(
                404,
                str(error),
                kind="invalid_request_error",
                code="model_not_found",
                param="model",
            ) from None
        except IdempotencyConflict as error:
            raise _HttpError(
                409,
                str(error),
                kind="invalid_request_error",
                code="idempotency_key_reused",
                param=IDEMPOTENCY_HEADER,
            ) from None
        try:
            done = await asyncio.shield(task)
        except Exception as error:  # noqa: BLE001 - every failure gets a canonical body
            raise _failure(error) from None
        return await self._deliver(
            request, access, done, stream=chat.stream, include_usage=chat.include_usage
        )

    async def _discard(
        self, request: web.Request, access: HttpModelAccess, producer: str
    ) -> web.Response:
        """``{"id": <completion id>}``: that completion no longer trains."""
        body = await _json(request)
        completion_id = body.get("id") if isinstance(body, dict) else None
        if not isinstance(completion_id, str) or not completion_id:
            raise _invalid("id must be the completion's id", code="invalid_value", param="id")
        try:
            access.discard(producer, completion_id)
        except AccessEnded:
            raise
        except UnknownGeneration as error:
            raise _HttpError(
                404,
                str(error),
                kind="invalid_request_error",
                code="generation_not_found",
                param="id",
            ) from None
        except ConversationBusy as error:
            raise _HttpError(
                409, str(error), kind="invalid_request_error", code="conversation_busy", param="id"
            ) from None
        except (ModelAccessError, ValueError) as error:
            raise _HttpError(
                409,
                str(error),
                kind="invalid_request_error",
                code="generation_not_discardable",
                param="id",
            ) from None
        return web.json_response(
            {"id": completion_id, "object": "loops.discard", "discarded": True}
        )

    async def _deliver(
        self,
        request: web.Request,
        access: HttpModelAccess,
        done: Completion | InvalidOutput,
        *,
        stream: bool,
        include_usage: bool,
    ) -> web.StreamResponse:
        """Write ``done`` whole, then settle it delivered; a 422 is a delivery too."""
        if isinstance(done, InvalidOutput):
            # An error is never a stream, as OpenAI's are not; the agent still got the
            # model's output, in ``error.reply``.
            status, stream, chunks = 422, False, [done.body()]
        else:
            status = 200
            chunks = done.events(include_usage=include_usage) if stream else [done.body()]
        response = web.StreamResponse(
            status=status,
            headers={
                "Content-Type": "text/event-stream" if stream else "application/json",
                "Cache-Control": "no-cache",
            },
        )
        await response.prepare(request)
        for chunk in chunks:
            await response.write(chunk)
        await response.write_eof()
        access.delivered(done.sample_id)
        return response


def _header(request: web.Request, name: str) -> str | None:
    values = request.headers.getall(name, [])
    if not values:
        return None
    if len(values) > 1 or not values[0] or len(values[0]) > _MAX_HEADER_VALUE:
        raise _invalid(f"{name} must be one nonempty value of at most 256 characters")
    return values[0]


async def _json(request: web.Request) -> object:
    try:
        return json.loads(await request.read())
    except web.HTTPRequestEntityTooLarge:
        raise _HttpError(
            413,
            f"The request body exceeds {MAX_REQUEST_BYTES} bytes",
            kind="invalid_request_error",
            code="request_too_large",
        ) from None
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise _invalid(f"The request body is not valid JSON: {error}") from None


def _path(location: tuple[int | str, ...]) -> str | None:
    """A validation location written as the policy writes params: ``messages[0].content``."""
    text = ""
    for part in location:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else str(part))
    return text or None


def _failure(error: Exception) -> _HttpError:
    """How a request that was admitted, but not served, is reported."""
    if isinstance(error, ContextWindowExceeded):
        return _invalid(str(error), code="context_length_exceeded", param="messages")
    if isinstance(error, UnsupportedParameterError):
        return _invalid(str(error), code="unsupported_parameter", param=error.param)
    if isinstance(error, InvalidRequestError):
        return _invalid(str(error), code="invalid_value", param=error.param)
    if isinstance(error, ConversationBusy):
        return _HttpError(
            409,
            str(error),
            kind="invalid_request_error",
            code="conversation_busy",
            param=CONVERSATION_HEADER,
        )
    if isinstance(error, (StaleRolloutError, InternalPolicyDeltaError)):
        return _HttpError(409, str(error), kind="invalid_request_error", code="rollout_stale")
    if isinstance(error, RenderingError):
        # Refused before sampling, so the attempt goes on; a 5xx would have stock
        # clients retry a request that renders the same way every time.
        return _invalid(str(error), code="rendering_failed", param="messages")
    logger.error("a generation failed: %s: %s", type(error).__name__, error, exc_info=error)
    return _server_error()


def _server_error() -> _HttpError:
    """A fault the agent cannot act on; its detail is in the run's log, not the body."""
    return _HttpError(
        500, "the gateway failed to serve this request", kind="server_error", code=None
    )
