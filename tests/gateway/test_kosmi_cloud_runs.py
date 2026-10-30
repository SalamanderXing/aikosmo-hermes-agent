"""AI Kosmo: Cloud runs (``/v1/runs/cloud``, ``/v1/runs/cloud/customer``) under the session authority.

Real HTTP API server, real authority execution, real agents and tool subprocesses; only the model wire is
synthetic. What is asserted is what reaches the model (tool schemas, system prompt, tool results) and
the tool processes, plus every file under HERMES_HOME.
"""
import asyncio
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time

KEY = "cloud-probe-owned-secret"
CUSTOMER_TOOLS = ["cloud_read", "cloud_read_attachment", "kosmo_cli"]


def test_cloud_runs_identity_customer_runtime_text_and_restart(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    work = tmp_path / "work"
    work.mkdir()
    state = tmp_path / "state"
    env = dict(os.environ, HERMES_HOME=str(state), HERMES_CWD=str(work), TERMINAL_CWD=str(work))
    # Phase one serves until it is killed mid-turn (a real restart: the process and its memory die).
    live = subprocess.Popen([sys.executable, __file__], cwd=work, env={**env, "PROBE_PHASE": "live"},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 240
    while not (state / "phase-live.json").exists():
        if live.poll() is not None or time.monotonic() > deadline:
            live.kill()
            raise AssertionError(live.communicate()[0])
        time.sleep(0.2)
    live.kill()
    live.wait()
    result = subprocess.run([sys.executable, __file__], cwd=work, env={**env, "PROBE_PHASE": "recover"},
                            capture_output=True, text=True, timeout=240, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((state / "cloud-receipt.json").read_text())
    assert receipt["env"] == {"alice": digest("alice-token"), "bob": digest("bob-token"), "ordinary": "absent"}
    assert receipt["customer_tools"] == CUSTOMER_TOOLS
    assert receipt["restart"] == {"without_credential": "failed", "with_credential": "completed"}
    assert repo.exists()


def test_text_snapshot_scrubs_thinking_and_caps_its_buffer():
    from gateway.platforms.api_server_text import MAX_TEXT_CHARS, RunText
    state = RunText()
    state.delta("<think>private reasoning</think>Hello")
    state.tool_started()
    assert (state.activity, state.text) == ("tool", "Hello")
    state.delta(" there")
    state.finish("completed", ttl=60)
    assert (state.text, state.status, state.activity) == ("Hello there", "completed", "thinking")
    big = RunText()
    big.delta("x" * (MAX_TEXT_CHARS + 1))
    big.finish("completed", ttl=60)
    assert (big.text, big.status) == ("", "unavailable"), "overflow stops streaming; the run itself goes on"


def test_orphaned_credentials_are_swept_and_pending_ones_kept(monkeypatch):
    """A Cloud request cancelled while its admission commits never reaches _execute_run or a refusal; its
    credential must not outlive the run sweep."""
    from types import SimpleNamespace
    from gateway import kosmi_cloud
    from gateway.platforms import api_server_authority_runs, api_server_runs
    credential = kosmi_cloud.CloudCredential("t", None, None)
    monkeypatch.setattr(kosmi_cloud, "_credentials", {})
    for run_id in ("young", "live-task", "queued", "settled", "never-committed"):
        kosmi_cloud.register(run_id, credential)
    old = time.monotonic() - kosmi_cloud.ORPHAN_GRACE_SECONDS - 1
    for run_id in ("live-task", "queued", "settled", "never-committed"):
        kosmi_cloud._credentials[run_id] = (old, credential)
    rows = {"queued": {"status": "queued"}, "settled": {"status": "terminal"}}
    monkeypatch.setattr(api_server_authority_runs, "run_admission",
                        lambda _adapter, run_id: (None, rows[run_id]) if run_id in rows else None)
    adapter = SimpleNamespace(_active_run_tasks={"live-task": object()})
    kosmi_cloud.sweep(lambda run_id: api_server_runs._run_pending(adapter, run_id))
    assert sorted(kosmi_cloud._credentials) == ["live-task", "queued", "young"]


def digest(token):
    import hashlib
    return hashlib.sha256(token.encode()).hexdigest()[:16]


async def start_gateway(instance):
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    runner = GatewayRunner()
    runner._running = True
    # On a restart this is boot recovery: lost turns are fenced unknown, unattended ones settle interrupted.
    authority = await initialize_session_authority(runner, profile_id="default", instance_id=instance)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": KEY, "port": 0}))
    adapter.gateway_runner = runner
    adapter.set_message_handler(runner._handle_message)
    runner.adapters[Platform.API_SERVER] = adapter
    assert await adapter.connect()
    return runner, authority, adapter, f"http://127.0.0.1:{adapter._site._server.sockets[0].getsockname()[1]}"


async def probe(peer):
    if os.environ["PROBE_PHASE"] == "recover":
        return await recover(peer)
    import aiohttp
    from gateway import kosmi_cloud

    home = Path(os.environ["HERMES_HOME"])
    os.environ["KOSMO_CLI_AUTH_ACCESS_TOKEN"] = "stale-global-token"  # process env must never be the identity
    _runner, authority, _adapter, base = await start_gateway("cloud-1")
    auth = {"Authorization": f"Bearer {KEY}"}

    def admissions():
        return authority.db._read_all("SELECT request_id, status, payload_json FROM session_admissions")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as client:
        async def start(path, body, headers=auth, expect=202):
            async with client.post(base + path, json=body, headers=headers) as response:
                data = await response.json()
                assert response.status == expect, (response.status, data)
                return data.get("run_id")

        async def finish(run_id):
            deadline = time.monotonic() + 60
            while True:
                async with client.get(f"{base}/v1/runs/{run_id}", headers=auth) as response:
                    status = await response.json()
                if status.get("status") in ("completed", "failed", "cancelled"):
                    return status
                assert time.monotonic() < deadline, status
                await asyncio.sleep(0.05)

        # Doors: the service bearer and the caller credential are both required; nothing is admitted.
        for path, body, headers, code in (
                ("/v1/runs/cloud", {"input": "x", "cloud_access_token": "t"}, {}, 401),
                ("/v1/runs/cloud", {"input": "x"}, auth, 401),
                ("/v1/runs/cloud", {"input": "x", "cloud_access_token": ""}, auth, 401),
                ("/v1/runs/cloud", {"input": "x", "cloud_access_token": {"bad": 1}}, auth, 401),
                ("/v1/runs/cloud/customer", {"input": "x", "cloud_access_token": "t"}, auth, 400),
                ("/v1/runs/cloud/customer", {"input": "x", "cloud_access_token": "t", "cloud_restricted_hotel_slugs": []}, auth, 400)):
            await start(path, body, headers, expect=code)
        assert admissions() == []

        # Two concurrent staff turns: each tool subprocess sees only its own caller's token.
        peer.hold = 2
        ids = await asyncio.gather(*(start("/v1/runs/cloud", {
            "session_id": f"staff-{name}", "input": f"ENV_PROBE {name}", "cloud_access_token": f"{name}-token"})
            for name in ("alice", "bob")))
        reader = asyncio.create_task(read_text(client, base, ids[0]))
        results = await asyncio.gather(*(finish(run_id) for run_id in ids))
        snapshots = await reader
        ordinary = await finish(await start("/v1/runs", {"session_id": "plain", "input": "ENV_PROBE ordinary"}))
        env = {name: seen(result) for name, result in zip(("alice", "bob", "ordinary"), (*results, ordinary))}
        # Text stream: thinking -> tool -> visible reply, closed with the run's status (streaming is off).
        assert snapshots[-1]["status"] == "completed" and f"SAW {digest('alice-token')}" in snapshots[-1]["text"], snapshots
        assert any(s["activity"] == "tool" for s in snapshots), snapshots
        assert all(set(s) == {"runId", "text", "status", "activity"} for s in snapshots)
        again = await read_text(client, base, ids[0])
        assert again[-1] == snapshots[-1], "a reconnecting reader gets the same snapshot"

        # An uploaded image survives into SessionDB as the image itself (Cloud history renders it).
        png = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
        pictured = await finish(await start("/v1/runs/cloud", {"session_id": "staff-image", "cloud_access_token": "alice-token",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "IMAGE_TURN look"},
                                                   {"type": "input_image", "image_url": f"data:image/png;base64,{png}"}]}]}))
        assert pictured["status"] == "completed", pictured
        user_rows = [m for m in authority.db.get_messages("staff-image") if m["role"] == "user"]
        stored = user_rows[-1]["content"]
        stored = json.loads(stored) if isinstance(stored, str) and stored.startswith("[") else stored
        assert isinstance(stored, list), user_rows
        assert any((p.get("image_url") or {}).get("url", p.get("image_url", "")).startswith("data:image/png")
                   if isinstance(p.get("image_url"), dict) else str(p.get("image_url", "")).startswith("data:image/png")
                   for p in stored), stored
        assert any("IMAGE_TURN" in str(p.get("text", "")) for p in stored), stored

        # The backend's personal-chat flow (apps/backend kosmi-sessions.ts hermesClient): its own session id,
        # a keyed run, then the transcript, listing and deletion through the session routes.
        async with client.post(base + "/api/sessions", headers=auth,
                               json={"id": "kosmi_flow", "source": "api_server"}) as response:
            assert response.status in (200, 201), await response.text()
            assert (await response.json())["session"]["id"] == "kosmi_flow"
        keyed = {**auth, "Idempotency-Key": "flow-1"}
        body = {"session_id": "kosmi_flow", "input": "FLOW_TURN hello", "cloud_access_token": "alice-token"}
        flow = await start("/v1/runs/cloud", body, keyed)
        assert await start("/v1/runs/cloud", body, keyed) == flow, "a keyed repeat returns the accepted run"
        assert (await finish(flow))["status"] == "completed"
        async with client.get(f"{base}/api/sessions/kosmi_flow/messages", headers=auth) as response:
            transcript = await response.json()
            assert response.status == 200, transcript
        # The backend projection reads only user/assistant/tool rows (hermes-messages.ts).
        chat = [(m["role"], m["content"]) for m in transcript["data"] if m["role"] in ("user", "assistant")]
        assert chat == [("user", "FLOW_TURN hello"), ("assistant", "LOCAL_ACK")], transcript
        async with client.get(f"{base}/api/sessions?limit=50", headers=auth) as response:
            assert any(row["id"] == "kosmi_flow" for row in (await response.json())["data"])
        async with client.delete(f"{base}/api/sessions/kosmi_flow", headers=auth) as response:
            assert response.status == 200, await response.text()

        # `@` references: a staff turn expands a workspace file; a customer turn never reads the workspace.
        Path("staff-private.txt").write_text("STAFF_PRIVATE_REF_SENTINEL")
        for lane, path, extra in (("customer", "/v1/runs/cloud/customer", {"cloud_restricted_hotel_slugs": ["my-hotel"]}),
                                  ("staff", "/v1/runs/cloud", {})):
            before = len(peer.requests)
            ref_run = await finish(await start(path, {"session_id": f"refs-{lane}", "cloud_access_token": "alice-token",
                                                      "input": f"REF_PROBE @file:{Path.cwd() / 'staff-private.txt'}", **extra}))
            assert ref_run["status"] == "completed", ref_run
            inlined = "STAFF_PRIVATE_REF_SENTINEL" in json.dumps([r["messages"] for r in peer.requests[before:]])
            stored = "STAFF_PRIVATE_REF_SENTINEL" in json.dumps(authority.db.get_messages(f"refs-{lane}"))
            assert (inlined, stored) == ((False, False) if lane == "customer" else (True, True)), (lane, inlined, stored)

        # Customer runtime: the effective tools and system prompt the model receives.
        upload = base64.b64encode(b"HTTP_UPLOAD_CONTENT").decode()
        customer_id = await start("/v1/runs/cloud/customer", {
            "session_id": "customer-one", "cloud_access_token": "carol-token", "cloud_restricted_hotel_slugs": ["my-hotel"],
            "instructions": "You are Kosmi. Help this customer use Cloud.",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "READ_UPLOAD please"},
                {"type": "input_image", "image_url": f"data:image/png;base64,{png}"},
                {"type": "input_file", "filename": "notes.txt", "file_data": "data:text/plain;base64," + upload}]}]})
        customer = await finish(customer_id)
        assert customer["status"] == "completed" and "HTTP_UPLOAD_CONTENT" in customer["output"], customer
        # The customer's image survives into SessionDB too (its agent's platform is cloud_customer).
        stored = next(m for m in authority.db.get_messages("customer-one") if m["role"] == "user")["content"]
        stored = json.loads(stored) if isinstance(stored, str) and stored.startswith("[") else stored
        assert "data:image/png" in json.dumps(stored), stored
        customer_requests = [r for r in peer.requests if "READ_UPLOAD" in json.dumps(r["messages"])]
        customer_tools = sorted({t["function"]["name"] for r in customer_requests for t in r.get("tools", [])})
        prompts = json.dumps([m for r in customer_requests for m in r["messages"] if m["role"] == "system"])
        assert "STAFF_" not in prompts and "Help this customer use Cloud" in prompts, prompts
        staff_prompts = json.dumps([m for r in peer.requests if "ENV_PROBE alice" in json.dumps(r["messages"])
                                    for m in r["messages"] if m["role"] == "system"])
        assert "STAFF_SOUL_SENTINEL" in staff_prompts, "control: a staff turn does see the staff SOUL"
        from gateway.session_contract import SessionRef
        agent = authority.agent(SessionRef("default", "customer-one"))
        assert agent.platform == "cloud_customer" and agent.max_iterations == 40
        assert agent._memory_enabled is False and agent._user_profile_enabled is False
        assert agent._memory_manager is None and agent.skip_background_review is True
        settings = {row["request_id"]: json.loads(row["payload_json"])["api_turn_v1"]["settings"]["cloud_runtime"]
                    for row in admissions()}
        assert settings[ids[0]] == "staff" and settings[customer_id] == "customer" and list(settings.values()).count(None) == 1
        assert kosmi_cloud._credentials == {}, "every credential leaves memory once its run ends"

        # Restart while a Cloud turn is running: it can never resume (its credential is gone).
        running = await start("/v1/runs/cloud", {"session_id": "staff-interrupted", "input": "BLOCK_FIFO long",
                              "cloud_access_token": "alice-token"})
        assert await asyncio.to_thread(peer.blocked.wait, 20)
        # Restart: two committed customer turns that never ran; the next process holds no credential.
        authority._schedule = lambda ref: None
        lost = await start("/v1/runs/cloud/customer", {"session_id": "customer-lost", "input": "LOST_TURN",
                           "cloud_access_token": "dave-token", "cloud_restricted_hotel_slugs": ["my-hotel"]})
        kept = await start("/v1/runs/cloud/customer", {"session_id": "customer-kept", "input": "KEPT_TURN",
                           "cloud_access_token": "erin-token", "cloud_restricted_hotel_slugs": ["my-hotel"]})
        (home / "phase-live.json").write_text(json.dumps({
            "env": env, "customer_tools": customer_tools, "running": running, "lost": lost, "kept": kept}))
        await asyncio.Event().wait()  # killed here, mid-turn


async def recover(peer):
    import aiohttp
    from gateway import kosmi_cloud
    from gateway.session_api_turn import recover_api_turns
    from tests.waiting import async_wait_until

    home = Path(os.environ["HERMES_HOME"])
    live = json.loads((home / "phase-live.json").read_text())
    running, lost, kept = live["running"], live["lost"], live["kept"]
    _runner, authority, adapter, base = await start_gateway("cloud-2")
    auth = {"Authorization": f"Bearer {KEY}"}
    assert kosmi_cloud._credentials == {}, "a new process holds no credential"
    # A credential for one of them (as if its caller were still bound): it runs restricted, from the
    # committed runtime, never as staff.
    kosmi_cloud.register(kept, kosmi_cloud.CloudCredential("erin-token", ("my-hotel",), None))
    recover_api_turns(adapter)
    interrupted = authority.db._read_all("SELECT status, outcome FROM session_admissions WHERE request_id=?", (running,))[0]
    assert (interrupted["status"], interrupted["outcome"]) == ("terminal", "interrupted"), dict(interrupted)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as client:
        # The interrupted conversation is not paused behind its lost turn: the next message runs.
        async with client.post(base + "/v1/runs/cloud", headers=auth, json={
                "session_id": "staff-interrupted", "input": "AFTER_RESTART", "cloud_access_token": "alice-token"}) as response:
            assert response.status == 202, await response.text()
            after = (await response.json())["run_id"]
        deadline = time.monotonic() + 60
        while True:
            async with client.get(f"{base}/v1/runs/{after}", headers=auth) as response:
                resumed = await response.json()
            if resumed.get("status") in ("completed", "failed", "cancelled"):
                break
            assert time.monotonic() < deadline, ("the conversation stayed paused behind its lost turn", resumed)
            await asyncio.sleep(0.05)
        assert resumed["status"] == "completed", resumed
    def terminal(run_id):
        rows = authority.db._read_all("SELECT status FROM session_admissions WHERE request_id=?", (run_id,))
        return rows and rows[0]["status"] == "terminal"
    await async_wait_until(lambda: terminal(lost) and terminal(kept), timeout=60, what="the recovered turns")
    outcome = {run_id: authority.db._read_all("SELECT outcome FROM session_admissions WHERE request_id=?", (run_id,))[0]["outcome"]
               for run_id in (lost, kept)}
    assert not any("LOST_TURN" in json.dumps(r["messages"]) for r in peer.requests), "a turn without its credential never ran"
    kept_tools = sorted({t["function"]["name"] for r in peer.requests if "KEPT_TURN" in json.dumps(r["messages"])
                         for t in r.get("tools", [])})
    assert kept_tools == CUSTOMER_TOOLS, kept_tools
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as client:
        # What the Cloud reconciler polls (kosmi-sessions.ts TERMINAL_RUN_STATUSES) for the turns the restart cut.
        async def polled(run_id):
            async with client.get(f"{base}/v1/runs/{run_id}", headers=auth) as response:
                assert response.status == 200, await response.text()
                return await response.json()
        cut, refused = await polled(running), await polled(lost)
        assert cut["status"] in ("cancelled", "interrupted", "failed"), cut
        assert refused["status"] == "failed" and "Send it again" in (refused.get("error") or ""), refused
    await adapter.disconnect()
    for file in home.rglob("*"):
        if file.is_file():
            content = file.read_bytes()
            for token in (b"alice-token", b"bob-token", b"carol-token", b"dave-token", b"erin-token"):
                assert token not in content, (file, token)
    (home / "cloud-receipt.json").write_text(json.dumps({
        "env": live["env"], "customer_tools": live["customer_tools"],
        "restart": {"without_credential": outcome[lost], "with_credential": outcome[kept]}}))


def seen(result):
    assert result["status"] == "completed", result
    return result["output"].removeprefix("SAW ").strip()


async def read_text(client, base, run_id):
    snapshots = []
    async with client.get(f"{base}/v1/runs/{run_id}/text", headers={"Authorization": f"Bearer {KEY}"}) as response:
        assert response.status == 200, await response.text()
        async for line in response.content:
            if line.startswith(b"data: "):
                snapshots.append(json.loads(line[6:]))
    return snapshots


def main():
    from http.server import ThreadingHTTPServer
    import re
    import threading
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(1, str(Path(__file__).parent / "fixtures"))
    from kosmi_peer import ScriptedPeer, newest_user_text, text, tool_call

    class ToolPeer(ScriptedPeer):
        """ENV_PROBE: a terminal call printing a digest of the caller token; READ_UPLOAD: the attachment
        reader on the staged @file; after a tool result: ``SAW <result>``. ``hold`` first requests wait for
        each other."""

        def script(self, request):
            last, user = request["messages"][-1], newest_user_text(request)
            if last["role"] == "tool":
                content = last["content"] if isinstance(last["content"], str) else json.dumps(last["content"])
                try:
                    content = json.loads(content).get("output") or json.loads(content).get("content") or content
                except (ValueError, AttributeError):
                    pass
                return text(f"SAW {content}")
            with self.server.lock:
                waiting = self.server.arrived < self.server.hold
                if waiting:
                    self.server.arrived += 1
            while waiting and self.server.arrived < self.server.hold and time.monotonic() < self.server.deadline:
                time.sleep(0.02)
            if "ENV_PROBE" in user:
                # A digest, never the token itself: the transcript keeps tool output.
                return tool_call("terminal", {"command": 'if [ -n "${KOSMO_CLI_AUTH_ACCESS_TOKEN+x}" ]; then printf "%s" '
                                              '"$KOSMO_CLI_AUTH_ACCESS_TOKEN" | sha256sum | cut -c1-16; else printf absent; fi'},
                                 f"call_{len(self.server.requests)}")
            if "READ_UPLOAD" in user:
                path = re.search(r"@file:([^\s\"\\]+)", user).group(1)
                return tool_call("cloud_read_attachment", {"path": path}, f"call_{len(self.server.requests)}")
            return text("LOCAL_ACK")

    peer = ThreadingHTTPServer(("127.0.0.1", 0), ToolPeer)
    peer.requests, peer.metadata_requests = [], []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    peer.lock, peer.hold, peer.arrived, peer.deadline = threading.Lock(), 0, 0, time.monotonic() + 120
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{peer.server_port}/v1"
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "SOUL.md").write_text("STAFF_SOUL_SENTINEL")
    (home / "memories").mkdir(exist_ok=True)
    (home / "memories" / "MEMORY.md").write_text("STAFF_MEMORY_SENTINEL")
    (home / "memories" / "USER.md").write_text("STAFF_USER_SENTINEL")
    Path("AGENTS.md").write_text("STAFF_REPO_SENTINEL")
    os.environ.update(OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=url)
    (home / "config.yaml").write_text(
        f"model:\n  default: local-wire-stub\n  provider: custom\n  base_url: {url}\n"
        "streaming:\n  enabled: false\nauxiliary:\n  title_generation:\n    enabled: false\n"
        "approvals:\n  mode: off\n")
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
