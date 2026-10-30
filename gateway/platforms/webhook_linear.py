"""AI Kosmo: the ``linear_comment`` webhook destination and the Linear staff lane (AIK-3110).

The final answer of a Linear comment delivery is posted as a Linear comment by the gateway, not by the
model: an accepted comment that gets no reply is the whole feature failing silently.

Where an answer goes is the destination upstream retains in each delivery's admission
(``native_text_v1.webhook_delivery``), so it survives a gateway restart. A one-shot delivery's chat
holds exactly that one admission (upstream ``retained_destination``). A shared thread chat
(``webhook:<route>:thread-<key>``, one session for every staff comment of one Linear thread) holds one
admission per comment: a comment's answer goes to its own delivery, keyed on its own delivery id; a turn
the gateway starts there itself (a T3 state event) answers where the thread's newest delivery answered,
under a fresh UUID v4. The authority runs a session's admissions one at a time in arrival order, so no
queue of our own orders the thread.

Exactly once, outbound: ``CommentCreateInput.id`` takes a client UUID v4, the delivery id (the
``Linear-Delivery`` header the edge maps onto ``X-Request-ID``). Before creating, and after any
ambiguous failure, the intended comment is read back and accepted only on a full match; a delivery
without a v4 id refuses to post rather than lose idempotency.
"""
import asyncio
import json
import logging
import os
import re
import urllib.request
import uuid
from contextvars import ContextVar
from typing import Any, Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)

LINEAR_API_URL = "https://api.linear.app/graphql"
LINEAR_API_TIMEOUT_SECONDS = 20
LINEAR_SILENCE_ACK = "Seen — nothing needed from me here."
_LINEAR_ID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
# CommentCreateInput.id must be UUID v4: version nibble 4, variant nibble 8/9/a/b.
_UUID4_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}")
_LINEAR_COMMENT_MUTATION = (
    "mutation KosmiCommentCreate($input: CommentCreateInput!) {"
    " commentCreate(input: $input) { success comment { id url } } }")
_LINEAR_COMMENT_READBACK = (
    "query KosmiCommentReadback($id: String!) {"
    " comment(id: $id) { id body parentId issue { id } } }")
# The destination fields a linear_comment delivery persists (validate_destination keeps only these).
DESTINATION_FIELDS = ("issue_id", "parent_id", "trigger")

# Reserved keys only a route SCRIPT may emit: stripped from the inbound body before the script runs.
# ``toolset_tiers``: names one of the route's toolset lists for this delivery.
TOOLSET_TIER_KEY = "__hermes_toolset_tier__"
# ``script_session_keys: true``: runs this delivery in the session every delivery with that key shares.
SESSION_KEY_KEY = "__hermes_session_key__"
RETRY_KEY = "__hermes_retry__"
_SESSION_KEY_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
THREAD_CHAT_PREFIX = "thread-"

# The turn whose final answer is being sent (``WebhookAdapter.send_final_ledgered``): its message id is
# the delivery id of a comment turn, or a producer identity for a turn the gateway started.
replying_turn: ContextVar = ContextVar("webhook_replying_turn", default=None)

_SIGNED_ROWS = """SELECT request_id, payload_json FROM session_admissions
    WHERE json_extract(payload_json, '$.native_text_v1.source.chat_id')=?
      AND json_extract(payload_json, '$.native_text_v1.source.platform')='webhook'
      AND json_extract(payload_json, '$.native_text_v1.automation') IS NULL
    ORDER BY seq"""


def route_tier_toolsets(route_config: dict, tier: Any) -> Optional[list[str]]:
    """The toolsets of a route's named tier, or None (the route's own ``toolsets``): an unknown tier, a
    route without ``toolset_tiers`` or an empty list narrows the run instead of widening it."""
    tiers = route_config.get("toolset_tiers")
    if not isinstance(tier, str) or not isinstance(tiers, dict) or not isinstance(tiers.get(tier), list):
        return None
    return [str(t).strip() for t in tiers[tier] if str(t).strip()] or None


def script_routing(route_config: dict, payload: Any) -> tuple[Optional[str], Optional[str]]:
    """``(toolset tier, shared-session key)`` a route script emitted for this delivery. Only a script
    can have put the reserved keys in ``payload`` (the inbound body is stripped before it runs)."""
    if not route_config.get("script") or not isinstance(payload, dict):
        return None, None
    tier = payload.get(TOOLSET_TIER_KEY)
    tier = tier if route_tier_toolsets(route_config, tier) else None
    key = payload.get(SESSION_KEY_KEY)
    if route_config.get("script_session_keys") is not True or not isinstance(key, str) or not _SESSION_KEY_RE.fullmatch(key):
        key = None
    return tier, key


def thread_chat_id(route_name: str, session_key: str) -> str:
    """Chat id of a route's shared (thread) session: the pre-v2 ``webhook:<route>:thread-<key>`` shape, so
    thread sessions started before v0.22 continue (the key is restricted to ``_SESSION_KEY_RE``)."""
    return f"webhook:{route_name}:{THREAD_CHAT_PREFIX}{session_key}"


def is_thread_chat(chat_id: str) -> bool:
    parts = (chat_id or "").split(":", 2)
    return len(parts) == 3 and parts[0] == "webhook" and parts[1] != "v2" and parts[2].startswith(THREAD_CHAT_PREFIX)


def delivery_uuid(chat_id: str) -> str:
    """The provider delivery id a one-shot ``webhook:v2:`` chat id names, else ''."""
    import base64
    import binascii
    if not (chat_id or "").startswith("webhook:v2:"):
        return ""
    token = chat_id[len("webhook:v2:"):]
    try:
        decoded = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
    except (binascii.Error, ValueError):
        return ""
    return decoded[2] if isinstance(decoded, list) and len(decoded) == 3 and isinstance(decoded[2], str) else ""


def _signed_rows(adapter, chat_id: str) -> list:
    from gateway.session_authorities import active_authority, serving_authorities
    runner = adapter.gateway_runner or getattr(adapter._message_handler, "__self__", None)
    scoped = active_authority(runner)
    candidates = [scoped] if scoped is not None else serving_authorities(runner)
    return [row for authority in candidates for row in authority.db._read_all(_SIGNED_ROWS, (chat_id,))]


def turn_destination(adapter, chat_id: str, turn=None) -> dict:
    """The validated destination of the turn answering on ``chat_id``; a linear_comment one also carries
    its ``reply_id``, the id the comment is created under. ``turn`` is the replying turn's event (default: ``replying_turn``)."""
    from gateway.platforms.webhook_delivery import checked_destination, retained_destination
    if not is_thread_chat(chat_id):
        return _with_reply_id(retained_destination(adapter, chat_id), delivery_uuid(chat_id))
    turn = turn if turn is not None else replying_turn.get()
    from gateway.session_authorities import all_authorities
    if not all_authorities(adapter.gateway_runner or getattr(adapter._message_handler, "__self__", None)):
        # No authority (standalone adapter): the process-local destination, as upstream's readback.
        from gateway.platforms.webhook_delivery import validate_destination
        return _with_reply_id(validate_destination(adapter._delivery_info.get(chat_id)),
                              str(getattr(turn, "message_id", "") or ""))
    rows = _signed_rows(adapter, chat_id)
    if not rows:
        from hermes_state_runtime import RuntimeStoreError
        raise RuntimeStoreError("not_found")
    message_id = str(getattr(turn, "message_id", "") or "")
    own = next((row for row in rows if row["request_id"] == message_id), None)
    row = own or rows[-1]
    destination = checked_destination(adapter, chat_id, json.loads(row["payload_json"]))
    # A gateway-started turn is one new answer: a reply id of its own.
    return _with_reply_id(destination, own["request_id"] if own else str(uuid.uuid4()))


def _with_reply_id(destination: dict, reply_id: str) -> dict:
    """A linear_comment destination carries the id its comment is created under; others are unchanged."""
    return {**destination, "reply_id": reply_id} if destination.get("deliver") == "linear_comment" else destination


def thread_tier(adapter, chat_id: str) -> Optional[str]:
    """A thread chat's toolset tier: its newest delivery's. The route script keys a shared session only
    for staff authors, so every delivery of one thread carries the same tier."""
    rows = _signed_rows(adapter, chat_id)
    delivery = json.loads(rows[-1]["payload_json"])["native_text_v1"].get("webhook_delivery") if rows else None
    return (delivery or {}).get("toolset_tier")


def silence_replacement(destination: dict) -> Optional[str]:
    """Answer anyway on a linear_comment destination whose gate owes the comment a reply. A comment that
    woke Kosmi only because the issue is assigned to it (``trigger: assigned``) stays silent."""
    if destination.get("deliver") != "linear_comment":
        return None
    if (destination.get("deliver_extra") or {}).get("trigger") == "assigned":
        return None
    return LINEAR_SILENCE_ACK


def _linear_graphql(query: str, variables: dict, api_key: str) -> dict:
    """POST one GraphQL request to Linear. Blocking: call via to_thread."""
    request = urllib.request.Request(
        LINEAR_API_URL, data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"content-type": "application/json", "authorization": api_key})
    with urllib.request.urlopen(request, timeout=LINEAR_API_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode())


def _comment_matches(comment: Any, comment_id: str, issue_id: str, parent_id: str, content: str) -> bool:
    return (isinstance(comment, dict) and comment.get("id") == comment_id
            and (comment.get("issue") or {}).get("id") == issue_id
            and (comment.get("parentId") or "") == parent_id
            and (comment.get("body") or "").strip() == content.strip())


async def _already_posted(comment_id: str, issue_id: str, parent_id: str, content: str, api_key: str) -> bool:
    """Read the intended comment back; True only on a full match (missing, another body or another thread
    answers False, so the caller fails rather than guessing)."""
    try:
        body = await asyncio.to_thread(_linear_graphql, _LINEAR_COMMENT_READBACK, {"id": comment_id}, api_key)
    except Exception as exc:
        logger.warning("[webhook] linear_comment read-back failed: %s", exc)
        return False
    if body.get("errors"):
        logger.warning("[webhook] linear_comment read-back rejected: %s", body["errors"])
        return False
    return _comment_matches((body.get("data") or {}).get("comment"), comment_id, issue_id, parent_id, content)


async def deliver_linear_comment(content: str, destination: dict) -> SendResult:
    """Post a final answer as a Linear comment keyed on the destination's ``reply_id``. Only a final send
    reaches here (``send`` drops non-notify sends); no ``message_id`` is returned, which keeps the stream
    consumer on its no-edit path so a tool boundary cannot post a second comment."""
    extra = destination.get("deliver_extra") or {}
    issue_id = str(extra.get("issue_id", "") or "").strip()
    parent_id = str(extra.get("parent_id", "") or "").strip()
    comment_id = str(destination.get("reply_id") or "")
    api_key = os.environ.get("LINEAR_API_KEY", "").strip()
    if not api_key:
        logger.error("[webhook] linear_comment delivery has no LINEAR_API_KEY")
        return SendResult(success=False, error="LINEAR_API_KEY is not set")
    if not _LINEAR_ID_RE.fullmatch(issue_id):
        logger.error("[webhook] linear_comment invalid issue_id: %r", issue_id)
        return SendResult(success=False, error="Invalid issue_id")
    if parent_id and not _LINEAR_ID_RE.fullmatch(parent_id):
        logger.error("[webhook] linear_comment invalid parent_id: %r", parent_id)
        return SendResult(success=False, error="Invalid parent_id")
    if not _UUID4_RE.fullmatch(comment_id):
        logger.error("[webhook] linear_comment has no UUIDv4 delivery id (%r): refusing to post without an "
                     "idempotency key", comment_id)
        return SendResult(success=False, error="Delivery id is not a UUIDv4")
    if await _already_posted(comment_id, issue_id, parent_id, content, api_key):
        logger.info("[webhook] Linear comment %s already posted — not reposting", comment_id)
        return SendResult(success=True)
    comment_input = {"id": comment_id, "issueId": issue_id, "body": content, **({"parentId": parent_id} if parent_id else {})}
    try:
        body = await asyncio.to_thread(_linear_graphql, _LINEAR_COMMENT_MUTATION, {"input": comment_input}, api_key)
    except Exception as exc:
        # Ambiguous: the server may have committed before the connection broke.
        if await _already_posted(comment_id, issue_id, parent_id, content, api_key):
            logger.info("[webhook] Linear comment %s had been committed", comment_id)
            return SendResult(success=True)
        logger.error("[webhook] linear_comment delivery error: %s", exc)
        return SendResult(success=False, error=str(exc))
    if body.get("errors") or not ((body.get("data") or {}).get("commentCreate") or {}).get("success"):
        # A duplicate-id rejection means an earlier attempt of ours won, which is success.
        detail = body.get("errors") or body
        if await _already_posted(comment_id, issue_id, parent_id, content, api_key):
            logger.info("[webhook] Linear comment %s already existed (%s)", comment_id, detail)
            return SendResult(success=True)
        logger.error("[webhook] linear_comment rejected: %s", detail)
        return SendResult(success=False, error=str(detail))
    logger.info("[webhook] Posted Linear comment %s on issue %s", comment_id, issue_id)
    return SendResult(success=True)
