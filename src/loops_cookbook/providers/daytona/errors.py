"""Classify a failed Daytona call by the SDK's and aiohttp's own error types.

Daytona's SDK raises one ``DaytonaError`` family that carries the HTTP status. It maps
transport failures to ``DaytonaConnectionError``, with the underlying aiohttp error as
the cause. These behaviours were observed against Daytona's API on 3 Oct 2026, with
SDK 0.220.0:

- A create over the per-sandbox limit answers ``400``, such as "CPU request 5 exceeds
  maximum allowed per sandbox (4)", and creates nothing.
- A second create under a taken name answers ``409``. Names are unique in the
  organization, so a create is idempotent by name.
- A deleted sandbox reads as ``destroying`` for about a second, and then ``404`` by
  id and by name. The name is free again at once, because Daytona renames the
  destroyed sandbox.

An organization limit is a rejection, but its status code has not been observed.
"""

from __future__ import annotations

import aiohttp
from daytona.common.errors import DaytonaBadRequestError, DaytonaConnectionError, DaytonaError

from loops_cookbook.resources.api import CreationDisposition

# A create that Daytona answered with one of these statuses left nothing. They mean a
# bad request, bad credentials, no permission or a rate limit. The SDK's create goes on
# calling Daytona after the create itself, so the status may belong to a later call.
# The adapter therefore looks the name up before it trusts the status. 409 is absent on
# purpose, because "already exists" means a sandbox does exist.
CREATE_LEFT_NOTHING = frozenset({400, 401, 403, 429})
# Statuses of a lookup that may succeed if repeated. They come from a just-created
# sandbox, or from the API shedding load.
RECONCILE_RETRY = frozenset({404, 429, 500, 502, 503, 504})
# Transport failures before a request was written, so nothing reached Daytona. The
# connector error's subclasses, such as DNS, proxy and TLS handshake errors, all fail
# while connecting.
UNDISPATCHED: tuple[type[BaseException], ...] = (
    aiohttp.ClientConnectorError,
    aiohttp.ConnectionTimeoutError,
)


def status_code(error: BaseException) -> int | None:
    """Return the HTTP status that a Daytona SDK error carries, if any."""
    return error.status_code if isinstance(error, DaytonaError) else None


def not_found(error: BaseException) -> bool:
    """Return whether Daytona answered ``404``."""
    return status_code(error) == 404


def rejected_before_sending(error: BaseException) -> bool:
    """Return whether the SDK or the facade refused a request before sending anything.

    Such a refusal comes from their own validation, as a ``DaytonaBadRequestError`` with
    no HTTP status.
    """
    return isinstance(error, DaytonaBadRequestError) and error.status_code is None


def never_dispatched(error: BaseException) -> bool:
    """Return true only when the request provably never left, because no connection was made.

    A read timeout, a reset mid-response, or a bare 5xx may each have created a
    sandbox, so only the connect-phase failures count.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, UNDISPATCHED):
            return True
        current = current.__cause__
    return False


def creation_disposition(error: BaseException) -> CreationDisposition:
    """Return what a failed create left, once a lookup by name found nothing.

    A ``400``, ``401``, ``403`` or ``429``, a request rejected before sending, and a
    request that never dispatched leave nothing. Anything else, including a ``409``, a
    timeout or a ``5xx``, may have left a sandbox that is not visible yet.
    """
    if (
        status_code(error) in CREATE_LEFT_NOTHING
        or rejected_before_sending(error)
        or never_dispatched(error)
    ):
        return CreationDisposition.DEFINITELY_NOT_CREATED
    return CreationDisposition.AMBIGUOUS


def create_refused(error: BaseException) -> bool:
    """Return whether a failed create is worth sending again.

    Only a rate limit and a request that never reached Daytona qualify. Each left nothing.
    """
    return status_code(error) == 429 or never_dispatched(error)


def reconcile_retryable(error: BaseException) -> bool:
    """Return whether a lookup of a just-created sandbox should be repeated."""
    return isinstance(error, (DaytonaConnectionError, TimeoutError, ConnectionError)) or (
        status_code(error) in RECONCILE_RETRY
    )
