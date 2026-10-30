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
    from tools.approval import is_session_yolo_enabled, disable_session_yolo

    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="delivery-test")
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": 0, "secret": _SECRET.decode(),
        "routes": {"auto-demo": {"prompt": "{text}", "durable_status": True, "yolo": True},
                   "plain": {"prompt": "{text}"}}}))
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

    def session_key(delivery_id):
        row = authority.db._read_all("SELECT target_session_id FROM session_admissions WHERE request_id=?", (delivery_id,))[0]
        return authority.sessions[row["target_session_id"]].route

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
            # A replayed turn (no admission-time state in this process) is armed by its own preparation.
            key = session_key("auto-demo-1-100")
            disable_session_yolo(key)
            authority._schedule = schedule
            for ref in held:
                schedule(ref)
            assert await asyncio.to_thread(peer.blocked.wait, 15)
            running = await status("auto-demo-1-100")
            assert running["status"] == "running" and running["started_at"] is not None and running["end_reason"] is None
            statuses.append(running["status"])
            assert is_session_yolo_enabled(key), "a yolo route's turn must never wait on an approval"
            peer.blocked.clear()
            peer.release.set()
            await async_wait_until(lambda: not adapter._background_tasks, timeout=30, what="the delivery to finalize")
            peer.release.clear()
            ended = await status("auto-demo-1-100")
            assert ended["status"] == "ended" and ended["end_reason"] == "completed", ended
            assert ended["ended_at"] is not None
            statuses.append(ended["status"])
            assert not is_session_yolo_enabled(key), "the close of the session releases the route's bypass"
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

            # Feedback runs in the delivery's own session, behind nothing, under the route's yolo.
            sid = ended["session_id"]
            feedback = asyncio.create_task(run_feedback(runner, "auto-demo-1-100", "BLOCK_FIFO make the hero bigger"))
            assert await asyncio.to_thread(peer.blocked.wait, 15)
            assert is_session_yolo_enabled(key)
            peer.release.set()
            plain = await feedback
            peer.release.clear()
            assert not is_session_yolo_enabled(key)
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
    from shared_authority_peer import ModelPeer
    peer = ThreadingHTTPServer(("127.0.0.1", 0), ModelPeer)
    peer.requests, peer.metadata_requests = [], []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{peer.server_port}/v1"
    os.environ.update(OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=url)
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(
        f"model:\n  default: local-wire-stub\n  provider: custom\n  base_url: {url}\n"
        "streaming:\n  enabled: false\nauxiliary:\n  title_generation:\n    enabled: false\n")
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
