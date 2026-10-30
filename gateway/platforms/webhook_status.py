"""AI Kosmo: status, transcript and operator feedback of a ``durable_status`` webhook delivery.

Read from the session authority's admission ledger, never a ledger of our own: a delivery is the
signed-ingress admission whose ``request_id`` is the provider delivery id (``session_admissions``,
committed before the HTTP 202). The wire keeps the values the auto-demo reconciler reads:
queued -> ``accepted``, started -> ``running``, terminal -> ``ended`` with the outcome as
``end_reason``, and unknown -> ``ended`` (``unknown_execution``): a turn lost across a restart is
paused and never finishes by itself. Timestamps are derived, never stored twice: ``accepted_at`` is
the delivery session's creation, ``started_at`` its first transcript row, ``ended_at`` the close
``finalize_webhook`` writes.
"""
import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The one route whose transcript and feedback are served (the Cloud demo-request page).
FEEDBACK_ROUTE = "auto-demo"
_WIRE_STATUS = {"queued": "accepted", "started": "running", "terminal": "ended", "unknown": "ended"}
_FEEDBACK_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
# Only the signed ingress admission is the delivery; later automation rows on its chat are not.
_DELIVERY_ROWS = """SELECT admission_id FROM session_admissions
    WHERE request_id=? AND principal_id LIKE 'messaging:%'
      AND json_extract(payload_json, '$.native_text_v1.source.platform')='webhook'
      AND json_extract(payload_json, '$.native_text_v1.automation') IS NULL"""


class FeedbackRefused(ValueError):
    """The feedback request cannot run; ``status`` is the HTTP answer."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def _webhook_adapter(runner):
    from gateway.config import Platform
    return (getattr(runner, "adapters", None) or {}).get(Platform.WEBHOOK)


def find_delivery(runner, delivery_id: str):
    """``(authority, admission row, route name)`` of one delivery on a ``durable_status`` route, else None
    (unknown id, a route that does not expose status, or an id two admissions share)."""
    from gateway.platforms.webhook import _WebhookDeliveryIdentity
    from gateway.session_authorities import serving_authorities
    from hermes_state_runtime import get_session_admission
    adapter = _webhook_adapter(runner)
    if adapter is None or not delivery_id:
        return None
    found = []
    for authority in serving_authorities(runner):
        for candidate in authority.db._read_all(_DELIVERY_ROWS, (delivery_id,)):
            row = get_session_admission(authority.db, admission_id=candidate["admission_id"])
            if row is None:
                continue
            route = _WebhookDeliveryIdentity.route_from_chat_id(row["payload"]["native_text_v1"]["source"]["chat_id"])
            config = adapter._routes.get(route)
            if isinstance(config, dict) and config.get("durable_status") is True:
                found.append((authority, row, route))
    return found[0] if len(found) == 1 else None


def delivery_status(runner, delivery_id: str) -> Optional[dict[str, Any]]:
    """Content-free status of a delivery (blocking SQLite reads: call off the loop)."""
    found = find_delivery(runner, delivery_id)
    if found is None:
        return None
    authority, row, route = found
    sid, status = row["target_session_id"], row["status"]
    session = authority.db.get_session(sid) or {}
    started_at = None
    if status != "queued":
        first = authority.db._read_all("SELECT MIN(timestamp) AS at FROM messages WHERE session_id=?", (sid,))
        started_at = first[0]["at"] if first else None
    ended = status in ("terminal", "unknown")
    return {
        "delivery_id": delivery_id, "route": route, "status": _WIRE_STATUS[status], "session_id": sid,
        "accepted_at": session.get("started_at"), "started_at": started_at,
        "ended_at": session.get("ended_at") if ended else None,
        "end_reason": row["outcome"] if status == "terminal" else "unknown_execution" if status == "unknown" else None,
    }


def delivery_transcript(runner, delivery_id: str):
    """``(session row, messages)`` of an auto-demo delivery's conversation (its compression tip), else None."""
    found = find_delivery(runner, delivery_id)
    if found is None or found[2] != FEEDBACK_ROUTE:
        return None
    authority = found[0]
    sid = authority.db.get_compression_tip(found[1]["target_session_id"]) or found[1]["target_session_id"]
    session = authority.db.get_session(sid)
    return None if session is None else (session, authority.db.get_messages(sid))


def _decode_image(part: dict) -> tuple[str, bytes]:
    from gateway.platforms.api_server import decode_data_url
    url = part.get("image_url")
    url = url.get("url") if isinstance(url, dict) else url
    mime, data = decode_data_url(url)
    if mime not in _FEEDBACK_IMAGE_TYPES:
        raise ValueError("unsupported_content_type:This image type is not supported.")
    return mime, data


def feedback_event(source, session_id: str, delivery_id: str, content: Any):
    """The feedback turn's event on the delivery's own chat: text parts, staged documents as ``@file``
    references (shared with the session chat door), images as native media the admission captures."""
    from gateway.platforms.api_server import (
        MAX_SESSION_ATTACHMENT_BYTES, _stage_session_file_parts, session_attachment_directory)
    from gateway.platforms.event import MessageEvent, MessageType
    parts = content if isinstance(content, list) else [{"type": "input_text", "text": content}]
    images = [_decode_image(part) for part in parts if isinstance(part, dict) and part.get("type") == "input_image"]
    if sum(len(data) for _, data in images) > MAX_SESSION_ATTACHMENT_BYTES:
        raise ValueError("invalid_content_part:Attached files exceed the 5 MB total limit.")
    staged = _stage_session_file_parts(
        [p for p in parts if not (isinstance(p, dict) and p.get("type") == "input_image")], session_id)
    text = "\n\n".join(p["text"] for p in staged if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"])
    if not text and not images:
        raise ValueError("missing_message:Feedback needs a message.")
    media = []
    if images:
        directory = session_attachment_directory(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        for mime, data in images:
            path = directory / f"{uuid.uuid4().hex[:10]}.{mime.split('/')[1]}"
            path.write_bytes(data)
            media.append((str(path), mime))
    return MessageEvent(
        text=text, message_type=MessageType.PHOTO if media else MessageType.TEXT, source=source,
        raw_message={"feedback_for": delivery_id}, message_id=f"{delivery_id}:feedback:{uuid.uuid4().hex}",
        timestamp=datetime.fromtimestamp(0, UTC), reply_expected=False,
        media_urls=[path for path, _ in media], media_types=[mime for _, mime in media])


async def run_feedback(runner, delivery_id: str, content: Any) -> dict[str, Any]:
    """Run one operator-feedback turn in an auto-demo delivery's original session and wait for it: a
    webhook admission on the delivery's chat with its retained destination, so the authority orders it
    behind the delivery's own turn and finalization closes the session again afterwards."""
    from gateway.session_envelope import restore_native
    from hermes_state_runtime import get_session_admission
    found = await asyncio.to_thread(find_delivery, runner, delivery_id)
    if found is None or found[2] != FEEDBACK_ROUTE:
        raise FeedbackRefused("Delivery not found", 404)
    authority, row, _route = found
    if row["status"] == "unknown":
        raise FeedbackRefused("The delivery's turn was lost in a restart; its session is paused", 409)
    adapter = _webhook_adapter(runner)
    envelope = row["payload"]["native_text_v1"]
    source = restore_native(row["payload"], runner).source
    try:
        event = await asyncio.to_thread(feedback_event, source, row["target_session_id"], delivery_id, content)
    except ValueError as exc:
        raise FeedbackRefused(str(exc).split(":", 1)[-1], 400) from exc
    # The one-shot session was closed by the delivery's finalization; a closed row would make the routing
    # store open a fresh conversation instead of continuing this one.
    await asyncio.to_thread(authority.db.reopen_session, row["target_session_id"])
    finalized = await adapter.admit_followup(event, dict(envelope["webhook_delivery"]))
    if finalized is not None:
        await asyncio.shield(finalized)
    rows = authority.db._read_all("SELECT admission_id FROM session_admissions WHERE request_id=?", (event.message_id,))
    settled = get_session_admission(authority.db, admission_id=rows[0]["admission_id"]) if rows else None
    return {"delivery_id": delivery_id, "status": settled["outcome"] if settled else "consumed"}

