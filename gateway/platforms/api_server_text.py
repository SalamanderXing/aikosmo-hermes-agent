"""AI Kosmo: ``GET /v1/runs/{run_id}/text``, public reply snapshots of a run for Cloud.

Each reader gets the latest accumulated reply text and the run's terminal status, so reconnects and
several tabs never consume each other's events (unlike the raw lifecycle queue). Only visible reply text
enters this transport: no tool, reasoning, credential or error fields. ``activity`` is a coarse phase for
the Cloud typing indicator: ``tool`` from a tool start until visible reply text arrives again
(``thinking``), because tool calls finish in milliseconds while the model keeps reasoning between them.
"""
import asyncio
import json
import logging
import time
from typing import Optional

from aiohttp import web

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 262_144


class RunText:
    """One run's snapshot; mutated on the event loop only (agent callbacks hop with call_soon_threadsafe)."""

    def __init__(self) -> None:
        from agent.memory_manager import StreamingContextScrubber
        from agent.think_scrubber import StreamingThinkScrubber
        self.text, self.version, self.status, self.activity = "", 0, "running", "thinking"
        self.changed = asyncio.Event()
        self.expires_at: Optional[float] = None
        self._think, self._context = StreamingThinkScrubber(), StreamingContextScrubber()

    def _bump(self) -> None:
        self.version += 1
        self.changed.set()

    def delta(self, delta: Optional[str]) -> None:
        """A raw model delta (``None`` closes a segment): scrubbed of thinking and injected context."""
        if delta is None:
            visible = self._context.feed(self._think.flush()) + self._context.flush()
            self._think.reset()
            self._context.reset()
        else:
            visible = self._context.feed(self._think.feed(delta))
        self._append(visible)

    def _append(self, visible: str) -> None:
        if self.status != "running" or not visible:
            return
        self.activity = "thinking"
        if len(self.text) + len(visible) > MAX_TEXT_CHARS:
            # Streaming stops; the run goes on and clients read the saved transcript.
            self.text, self.status = "", "unavailable"
            logger.warning("[api_server] run text buffer exceeded its limit")
        else:
            self.text += visible
        self._bump()

    def tool_started(self) -> None:
        if self.status == "running" and self.activity != "tool":
            self.activity = "tool"
            self._bump()

    def finish(self, status: str, ttl: float) -> None:
        self._append(self._context.feed(self._think.flush()) + self._context.flush())
        if self.status != "unavailable":
            self.status = status
        self.expires_at = time.time() + ttl
        self._bump()


async def handle_run_text(adapter, request: web.Request) -> web.StreamResponse:
    """Replay the latest public text snapshot to each independent reader."""
    if auth_err := adapter._check_auth(request):
        return auth_err
    run_id = request.match_info["run_id"]
    if not adapter._request_owns_run(request, run_id):
        return web.json_response({"error": "Run not found"}, status=404)
    state = adapter._run_text_streams.get(run_id)
    if state is None:
        return web.json_response({"error": "Text stream unavailable"}, status=404)
    response = web.StreamResponse(headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})
    await response.prepare(request)
    version = -1
    try:
        while True:
            if version != state.version:
                version = state.version
                payload = json.dumps({"runId": run_id, "text": state.text, "status": state.status,
                                      "activity": state.activity})
                await response.write(f"event: text\nid: {version}\nretry: 1000\ndata: {payload}\n\n".encode())
            if state.status != "running":
                break
            state.changed.clear()
            try:
                await asyncio.wait_for(state.changed.wait(), timeout=10)
            except TimeoutError:
                await response.write(b": keepalive\n\n")
            await asyncio.sleep(0.05)
    except ConnectionError:
        logger.debug("[api_server] text reader disconnected: %s", run_id)
    return response


def sweep(adapter, now: float) -> None:
    for run_id, state in list(adapter._run_text_streams.items()):
        if state.expires_at is not None and now > state.expires_at:
            del adapter._run_text_streams[run_id]
            adapter._release_run_owner_if_forgotten(run_id)
