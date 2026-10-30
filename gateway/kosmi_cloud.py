"""AI Kosmo: per-run Cloud caller credentials and the customer runtime, under the session authority.

``/v1/runs/cloud`` (staff) and ``/v1/runs/cloud/customer`` turns run like every API turn: the HTTP
request admits the turn and the authority's executor runs it (``gateway.session_ingress``). What the
caller sends that must never be durable (the Cloud JWT, the customer's hotel slugs, the upload
directory its reader may open) lives only in this process: :func:`register` stores it before the
admission commits, keyed by the admission's request id (the run id), and :func:`cloud_turn` binds it
around the turn through ContextVars, never ``os.environ``, then drops it. Which runtime the turn gets
is not secret and is committed with the admission (``settings.cloud_runtime``), so a turn replayed
after a restart is still a customer turn; it finds no credential and fails closed instead of running
with the shared operator identity, and one that was running when the process died settles interrupted at
boot (``gateway.session_unattended_unknown``) instead of pausing its conversation.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

CLOUD_RUNTIMES = ("staff", "customer")
CUSTOMER_PLATFORM = "cloud_customer"
CUSTOMER_TOOLSET = "kosmo-cloud-customer"
CUSTOMER_TOOLS = frozenset({"cloud_read", "cloud_read_attachment", "kosmo_cli"})
# kosmo-cli work is multi-step (help, read, change, verify).
CUSTOMER_MAX_ITERATIONS = 40
CREDENTIAL_LOST = ("This Cloud message was saved but Hermes restarted before it ran, and the sign-in it "
                   "carried is not kept across restarts. Send it again.")


@dataclass(frozen=True)
class CloudCredential:
    token: str = field(repr=False)
    hotel_slugs: Optional[tuple[str, ...]] = None  # customer runs: the only properties kosmo-cli may touch
    attachment_root: Optional[str] = None  # customer runs: the one upload directory the reader may open


_credentials: dict[str, CloudCredential] = {}
_lock = threading.Lock()


def register(request_id: str, credential: CloudCredential) -> None:
    with _lock:
        _credentials[request_id] = credential


def discard(request_id: str) -> None:
    with _lock:
        _credentials.pop(request_id, None)


def cloud_runtime(prepared) -> Optional[str]:
    """The committed runtime of an executing API admission (``api_execution``), else None."""
    settings = (prepared or {}).get("settings") or {}
    runtime = settings.get("cloud_runtime")
    return runtime if runtime in CLOUD_RUNTIMES else None


def is_cloud_admission(payload) -> bool:
    """Whether a committed API admission is a Cloud turn (its runtime is committed with it)."""
    return cloud_runtime(payload.get("api_turn_v1")) is not None


def customer_turn() -> bool:
    """Whether the executing turn is a customer Cloud turn (the restricted runtime)."""
    from gateway.session_api_turn import api_execution
    return cloud_runtime(api_execution.get()) == "customer"


@contextmanager
def cloud_turn(row, prepared) -> Iterator[Optional[str]]:
    """Bind a Cloud admission's caller credential for its turn. Yields None to run the turn, or the
    refusal to answer instead (a Cloud turn with no credential in this process). A turn that is not a
    Cloud turn binds nothing."""
    if cloud_runtime(prepared) is None:
        yield None
        return
    with _lock:
        credential = _credentials.get(row["request_id"]) if row.get("principal_id") == "api" else None
    if credential is None:
        yield CREDENTIAL_LOST
        return
    from gateway.session_context import (
        kosmo_attachment_root, kosmo_cli_access_token, kosmo_cli_restricted_hotel_slugs)
    tokens = [(var, var.set(value)) for var, value in (
        (kosmo_cli_access_token, credential.token),
        (kosmo_cli_restricted_hotel_slugs, credential.hotel_slugs),
        (kosmo_attachment_root, credential.attachment_root))]
    try:
        yield None
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
        discard(row["request_id"])


def check_customer_agent(agent) -> None:
    """Construction fails closed when the customer agent's effective tools differ from the allowlist
    (automatic tool discovery or a context engine must not widen it)."""
    if set(agent.valid_tool_names) != CUSTOMER_TOOLS:
        raise RuntimeError("Customer Kosmi tool allowlist mismatch")
