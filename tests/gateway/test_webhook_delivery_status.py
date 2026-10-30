"""AI Kosmo: delivery status, route yolo and operator feedback, read from the authority's admission ledger.

Real signed HTTP ingress, a real session authority and real turns against a loopback model; only the
provider wire is synthetic (``fixtures/shared_authority_peer.ModelPeer``).
"""
import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

_SECRET = b"owned-hook-secret"


def test_delivery_status_only_serves_only_status_routes():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"delivery_status_only": True}))
    routes = {(method, path) for method, path, _ in adapter._http_route_table()}
    assert ("GET", "/api/webhook-deliveries/{delivery_id}") in routes
    assert ("GET", "/api/webhook-deliveries/{delivery_id}/session") in routes
    assert ("POST", "/api/webhook-deliveries/{delivery_id}/feedback") in routes
    assert ("POST", "/v1/runs") not in routes and ("POST", "/api/sessions") not in routes
    full = {(m, p) for m, p, _ in APIServerAdapter(PlatformConfig(enabled=True, extra={}))._http_route_table()}
    assert ("POST", "/v1/runs") in full and ("GET", "/api/webhook-deliveries/{delivery_id}") in full


def test_delivery_lifecycle_yolo_and_feedback_through_the_authority(tmp_path):
    result = subprocess.run([sys.executable, __file__],
        env=dict(os.environ, HERMES_HOME=str(tmp_path)), capture_output=True, text=True, timeout=180, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / "delivery-receipt.json").read_text())
    assert receipt["statuses"] == ["accepted", "running", "ended"]
    assert receipt["feedback"] == {"plain": "completed", "attachments": "completed"}


async def probe(peer):
    import aiohttp
    from aiohttp.test_utils import make_mocked_request
    from gateway.config import Platform
    from gateway.platforms.webhook import WebhookAdapter
    from gateway.platforms.webhook_status import delivery_status, delivery_transcript, run_feedback, FeedbackRefused
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from tests.waiting import async_wait_until
    from tools import approval

    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="delivery-test")
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": 0, "secret": _SECRET.decode(),
        "routes": {"auto-demo": {"prompt": "{text}", "durable_status": True, "yolo": True, "toolsets": ["terminal"]},
                   "plain": {"prompt": "{text}", "toolsets": ["terminal"]}}}))
    adapter.gateway_runner = runner
    runner.adapters[Platform.WEBHOOK] = adapter
    runner._wire_adapter_handlers(adapter)
    api = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    api.gateway_runner = runner
    assert await adapter.connect()
    port = next(iter(adapter._runner.sites))._server.sockets[0].getsockname()[1]
    statuses = []

    async def post(client, route, text, delivery_id):
        body = json.dumps({"text": text}).encode()
        response = await client.post(f"http://127.0.0.1:{port}/webhooks/{route}", data=body, headers={
            "X-Request-ID": delivery_id, "X-Hub-Signature-256": "sha256=" + hmac.new(_SECRET, body, hashlib.sha256).hexdigest()})
        assert response.status == 202, (response.status, await response.text())
        assert (await response.json())["delivery_id"] == delivery_id

    async def status(delivery_id):
        return await asyncio.to_thread(delivery_status, runner, delivery_id)

    try:
        async with aiohttp.ClientSession() as client:
            # Queued: committed, not yet claimed (execution held at scheduling).
            schedule = authority._schedule
            held = []
            authority._schedule = held.append
            await post(client, "auto-demo", "BLOCK_FIFO first", "auto-demo-1-100")
            queued = await status("auto-demo-1-100")
            assert queued["status"] == "accepted" and queued["started_at"] is None and queued["ended_at"] is None
            assert queued["accepted_at"] is not None and queued["route"] == "auto-demo"
            statuses.append(queued["status"])
            authority._schedule = schedule
            for ref in held:
                schedule(ref)
            assert await asyncio.to_thread(peer.blocked.wait, 15)
            running = await status("auto-demo-1-100")
            assert running["status"] == "running" and running["started_at"] is not None and running["end_reason"] is None
            statuses.append(running["status"])
            peer.blocked.clear()
            peer.release.set()
            await async_wait_until(lambda: not adapter._background_tasks, timeout=30, what="the delivery to finalize")
            peer.release.clear()
            ended = await status("auto-demo-1-100")
            assert ended["status"] == "ended" and ended["end_reason"] == "completed", ended
            assert ended["ended_at"] is not None
            statuses.append(ended["status"])
            assert set(ended) == {"delivery_id", "route", "status", "session_id", "accepted_at", "started_at",
                                  "ended_at", "end_reason"}
            assert "BLOCK_FIFO" not in json.dumps(ended)

            # The HTTP door: auth and the same body.
            response = await api._handle_webhook_delivery(make_mocked_request(
                "GET", "/", match_info={"delivery_id": "auto-demo-1-100"}))
            assert response.status == 200 and json.loads(response.text) == ended

            # Not exposed: a route without durable_status, and an unknown id.
            await post(client, "plain", "hello", "plain-1")
            await async_wait_until(lambda: not adapter._background_tasks, timeout=30, what="the plain delivery")
            assert await status("plain-1") is None and await status("missing") is None

            # Feedback runs in the delivery's own session, behind nothing.
            sid = ended["session_id"]
            peer.blocked.clear()
            feedback = asyncio.create_task(run_feedback(runner, "auto-demo-1-100", "BLOCK_FIFO make the hero bigger"))
            assert await asyncio.to_thread(peer.blocked.wait, 15)
            peer.release.set()
            plain = await feedback
            peer.release.clear()
            session, messages = await asyncio.to_thread(delivery_transcript, runner, "auto-demo-1-100")
            assert session["id"] == sid and session["ended_at"] is not None
            users = [m["content"] for m in messages if m["role"] == "user"]
            assert any("make the hero bigger" in str(c) for c in users), users
            assert sum(1 for m in messages if m["role"] == "assistant") >= 2

            png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
            attached = await run_feedback(runner, "auto-demo-1-100", [
                {"type": "input_text", "text": "use this logo and these notes"},
                {"type": "input_image", "image_url": f"data:image/png;base64,{png}"},
                {"type": "input_file", "filename": "notes.txt", "file_data": "data:text/plain;base64,aGVsbG8="}])
            request = json.dumps(peer.requests[-1]["messages"])
            assert "use this logo" in request and "@file:" in request and "notes.txt" in request
            assert "image_url" in request or "image" in request
            # The delivery's own destination still resolves with feedback turns on its chat.
            from gateway.platforms.webhook_delivery import retained_destination
            chat = authority.sessions[sid].source.chat_id
            assert retained_destination(adapter, chat)["deliver"] == "log"

            # Feedback racing the delivery's own close: finalization is released between the reopen and
            # the admission; the feedback must still land in the delivery's session.
            gate, real_end = asyncio.Event(), adapter._end_webhook_session

            async def gated_end(event, chat_id):
                await gate.wait()
                await real_end(event, chat_id)
            adapter._end_webhook_session = gated_end
            await post(client, "auto-demo", "second run", "auto-demo-2-200")
            await async_wait_until(lambda: (delivery_status(runner, "auto-demo-2-200") or {}).get("status") == "ended",
                                   timeout=30, what="the second delivery to settle")
            second_sid = delivery_status(runner, "auto-demo-2-200")["session_id"]
            real_reopen, loop = authority.db.reopen_session, asyncio.get_running_loop()

            def reopen_then_release_close(session_id):
                real_reopen(session_id)
                loop.call_soon_threadsafe(gate.set)
                time.sleep(0.3)  # without the close lock, finalization closes the session right here
            authority.db.reopen_session = reopen_then_release_close
            raced = await run_feedback(runner, "auto-demo-2-200", "race feedback")
            authority.db.reopen_session, adapter._end_webhook_session = real_reopen, real_end
            assert raced["status"] == "completed"
            (row,) = authority.db._read_all("SELECT target_session_id FROM session_admissions WHERE request_id LIKE ?",
                                            ("auto-demo-2-200:feedback:%",))
            assert row["target_session_id"] == second_sid, "feedback rotated into a fresh session"

            # A feedback request cancelled while the authority reserves its route, with the delivery's own
            # close pending: the admission still owns the session, so the close waits for it, and this and
            # the next feedback stay in the delivery's session.
            import threading
            from gateway.platforms.webhook_status import feedback_event, find_delivery
            from gateway.session_envelope import restore_native
            close_gate = asyncio.Event()

            async def held_end(event, chat_id):
                await close_gate.wait()
                await real_end(event, chat_id)
            adapter._end_webhook_session = held_end
            await post(client, "auto-demo", "third run", "auto-demo-3-300")
            await async_wait_until(lambda: (delivery_status(runner, "auto-demo-3-300") or {}).get("status") == "ended",
                                   timeout=30, what="the third delivery to settle")
            _authority, delivered, _route = find_delivery(runner, "auto-demo-3-300")
            third_sid = delivered["target_session_id"]
            source = restore_native(delivered["payload"], runner).source
            destination = dict(delivered["payload"]["native_text_v1"]["webhook_delivery"])
            entered, proceed = threading.Event(), threading.Event()
            real_reserve = authority._reserve_route

            def held_reserve(src):
                entered.set()
                assert proceed.wait(20)
                return real_reserve(src)
            authority._reserve_route = held_reserve
            cancelled = asyncio.create_task(adapter.admit_followup(
                feedback_event(source, third_sid, "auto-demo-3-300", "cancelled feedback"), destination,
                lambda: authority.db.reopen_session(third_sid)))
            assert await asyncio.to_thread(entered.wait, 15)
            cancelled.cancel()
            try:
                await cancelled
            except asyncio.CancelledError:
                pass
            close_gate.set()
            await asyncio.sleep(0.3)  # without an owned critical section, the close lands here
            proceed.set()
            authority._reserve_route, adapter._end_webhook_session = real_reserve, real_end
            await async_wait_until(lambda: not adapter._background_tasks, timeout=30, what="the cancelled feedback")
            await run_feedback(runner, "auto-demo-3-300", "after a cancelled request")
            targets = {r["target_session_id"] for r in authority.db._read_all(
                "SELECT target_session_id FROM session_admissions WHERE request_id LIKE ?", ("auto-demo-3-300%",))}
            assert targets == {third_sid}, targets

            # Route yolo, as behavior: a dangerous (not hardline) command runs in a yolo route's turns, a
            # replayed turn and a feedback turn included, and waits for an approval nobody gives (denied on
            # timeout) on any other route. The per-session bypass registry is never touched.
            home = Path(os.environ["HERMES_HOME"])
            doomed = {name: home / f"yolo-{name}" for name in ("replayed", "feedback", "plain")}
            for directory in doomed.values():
                directory.mkdir()
            held = []
            authority._schedule = held.append
            await post(client, "auto-demo", f"DANGER {doomed['replayed']}", "auto-demo-6-600")
            authority._schedule = schedule
            for ref in held:
                schedule(ref)
            await async_wait_until(lambda: not doomed["replayed"].exists(), timeout=60,
                                   what="the yolo route's replayed turn to run its command")
            assert (await run_feedback(runner, "auto-demo-6-600", f"DANGER {doomed['feedback']}"))["status"] == "completed"
            assert not doomed["feedback"].exists(), "a feedback turn on a yolo route waited on an approval"
            await post(client, "plain", f"DANGER {doomed['plain']}", "plain-2")
            await async_wait_until(lambda: not adapter._background_tasks, timeout=60, what="the plain route's turn")
            assert doomed["plain"].exists(), "a route without yolo ran a command that needs approval"
            assert not approval._session_yolo, "route yolo is turn-scoped, never the global bypass"
            for refused, code in (("missing", 404), ("plain-1", 404)):
                try:
                    await run_feedback(runner, refused, "x")
                except FeedbackRefused as exc:
                    assert exc.status == code
                else:
                    raise AssertionError(f"feedback on {refused} was accepted")
            Path(os.environ["HERMES_HOME"], "delivery-receipt.json").write_text(json.dumps({
                "statuses": statuses, "feedback": {"plain": plain["status"], "attachments": attached["status"]}}))
    finally:
        peer.release.set()
        await adapter.disconnect()


def main():
    from http.server import ThreadingHTTPServer
    import threading
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(1, str(Path(__file__).parent / "fixtures"))
    from kosmi_peer import ScriptedPeer, newest_user_text, tags_or_ack, text, tool_call

    class DangerPeer(ScriptedPeer):
        """``DANGER <dir>``: a terminal call deleting it (needs approval, not hardline); then ``TOOL_DONE``."""

        def script(self, request):
            if request["messages"][-1]["role"] == "tool":
                return text("TOOL_DONE")
            target = re.search(r'DANGER (/[^\s"\\]+)', newest_user_text(request))
            return tool_call("terminal", {"command": f"rm -rf {target.group(1)}"}) if target else tags_or_ack(request)

    peer = ThreadingHTTPServer(("127.0.0.1", 0), DangerPeer)
    peer.requests, peer.metadata_requests = [], []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{peer.server_port}/v1"
    os.environ.update(OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=url)
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(
        f"model:\n  default: local-wire-stub\n  provider: custom\n  base_url: {url}\n"
        "streaming:\n  enabled: false\nauxiliary:\n  title_generation:\n    enabled: false\n"
        "approvals:\n  timeout: 3\n")
    try:
        asyncio.run(probe(peer))
    finally:
        peer.shutdown()
        peer.server_close()


if __name__ == "__main__":
    import traceback
    status = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
