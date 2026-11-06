"""How a failed Daytona call is classified, by the SDK's and aiohttp's own error types.

Daytona's SDK raises one ``DaytonaError`` family that carries the HTTP status, and
maps transport failures to ``DaytonaConnectionError`` with the underlying aiohttp
error as its cause. Observed against Daytona's API on 3 Oct 2026, SDK 0.220.0:

- a sandbox over the per-sandbox limit answers ``400`` ("CPU request 5 exceeds
  maximum allowed per sandbox (4)"), and nothing is created;
- a second create under a taken name answers ``409``: names are unique in the
  organization, so a create is idempotent by name;
- a deleted sandbox reads as ``destroying`` for about a second, then ``404`` by id
  and by name, and the name is free again at once (Daytona renames the destroyed one).

An organization limit is a rejection; its code is unobserved.
"""

from __future__ import annotations

import aiohttp
from daytona.common.errors import DaytonaBadRequestError, DaytonaConnectionError, DaytonaError

from loops_cookbook.resources.api import CreationDisposition

# A create Daytona answered with one of these (bad request, credentials, permission,
# or a rate limit) left nothing. The SDK's create goes on calling Daytona after the
# create itself, so the status may be a later call's: the adapter looks the name up
# before taking it as the create's. 409 is deliberately absent: "already exists"
# means one does.
CREATE_LEFT_NOTHING = frozenset({400, 401, 403, 429})
# A lookup that may succeed if repeated: a just-created sandbox, or the API shedding load.
RECONCILE_RETRY = frozenset({404, 429, 500, 502, 503, 504})
# Transport failures before a request was written: nothing reached Daytona. The
# connector error's subclasses (DNS, proxy, TLS handshake) are all connect-phase.
UNDISPATCHED: tuple[type[BaseException], ...] = (
    aiohttp.ClientConnectorError,
    aiohttp.ConnectionTimeoutError,
)


def status_code(error: BaseException) -> int | None:
    """The HTTP status a Daytona SDK error carries, if any."""
    return error.status_code if isinstance(error, DaytonaError) else None


def not_found(error: BaseException) -> bool:
    """Whether Daytona answered ``404``."""
    return status_code(error) == 404


def rejected_before_sending(error: BaseException) -> bool:
    """Whether the SDK, or the facade, refused a request before sending anything: its
    own validation, a ``DaytonaBadRequestError`` with no HTTP status."""
    return isinstance(error, DaytonaBadRequestError) and error.status_code is None


def never_dispatched(error: BaseException) -> bool:
    """True only when the request provably never left: the connection was never made.

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
    """What a failed create left, once a lookup by name found nothing.

    ``400``, ``401``, ``403``, ``429``, a request rejected before sending, and a
    request that never dispatched leave nothing; everything else, including ``409``, a
    timeout, and a ``5xx``, may have left a sandbox that is not visible yet.
    """
    if (
        status_code(error) in CREATE_LEFT_NOTHING
        or rejected_before_sending(error)
        or never_dispatched(error)
    ):
        return CreationDisposition.DEFINITELY_NOT_CREATED
    return CreationDisposition.AMBIGUOUS


def create_refused(error: BaseException) -> bool:
    """Whether a failed create is worth sending again: a rate limit, or a request that
    never reached Daytona. Each left nothing."""
    return status_code(error) == 429 or never_dispatched(error)


def reconcile_retryable(error: BaseException) -> bool:
    """Whether a lookup of a just-created sandbox should be repeated."""
    return isinstance(error, (DaytonaConnectionError, TimeoutError, ConnectionError)) or (
        status_code(error) in RECONCILE_RETRY
    )
