"""AI Kosmo: the linear_comment destination and the Linear staff lane on the session authority.

Real signed HTTP ingress, a real route script, real admissions and turns against a loopback model, and a
gateway restart onto the same state.db. Only the model wire and Linear's GraphQL endpoint are synthetic.
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

import pytest

_SECRET = b"lane-secret"
ROOT = "c0000000-0000-4000-8000-000000000001"
ISSUE = "a0000000-0000-4000-8000-0000000000aa"
OTHER_ISSUE = "a0000000-0000-4000-8000-0000000000bb"
D = {n: f"{n}0000000-0000-4000-8000-00000000000{n}" for n in "123456789"}
GATE = """import json, sys
payload = json.load(sys.stdin)
if payload.get("retry"):
    print(json.dumps({"__hermes_retry__": True}))
    raise SystemExit(0)
out = dict(payload)
if payload.get("author") == "staff":
    out["__hermes_toolset_tier__"] = "staff"
    out["__hermes_session_key__"] = payload["thread"]
print(json.dumps(out))
"""
ROUTE = {"durable_status": True, "script": "gate.py", "toolsets": ["todo"],
         "toolset_tiers": {"staff": ["todo", "memory"]}, "script_session_keys": True, "prompt": "{text}",
         "deliver": "linear_comment",
         "deliver_extra": {"issue_id": "{issue}", "parent_id": "{parent}", "trigger": "{trigger}"}}


def test_linear_lane_through_the_authority_and_a_restart(tmp_path):
    result = subprocess.run([sys.executable, __file__],
        env=dict(os.environ, HERMES_HOME=str(tmp_path)), capture_output=True, text=True, timeout=240, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / "lane-receipt.json").read_text())
    assert receipt["thread_order"] == [D["1"], D["2"]]
    assert receipt["restart_reply"] == D["6"]


def _thread_payload(trigger):
    return {"issue": ISSUE, "parent": ROOT, "trigger": trigger,
            "__hermes_toolset_tier__": "staff", "__hermes_session_key__": ROOT}


async def _admit(adapter, delivery_id, trigger):
    return await adapter._admit_agent_run(
        _thread_payload(trigger), "comment", delivery_id, 0.0, route_config=ROUTE, route_name="linear-comment",
        profile=None, event_type="Comment")


@pytest.mark.asyncio
async def test_concurrent_thread_deliveries_each_commit_their_own_destination():
    """Two comments of one thread admitted while a third admission holds the lock: each keeps its trigger
    (the destination is published and snapshotted inside the same critical section)."""
    _runner, authority, adapter, _port, _tiers = await gateway(None, [])
    authority._schedule = lambda ref: None
    try:
        await adapter._session_close_lock.acquire()
        first = asyncio.create_task(_admit(adapter, D["1"], "mention"))
        await asyncio.sleep(0)
        second = asyncio.create_task(_admit(adapter, D["2"], "assigned"))
        await asyncio.sleep(0)
        adapter._session_close_lock.release()
        assert await asyncio.gather(first, second) == ["accepted", "accepted"]
        triggers = {row["request_id"]: json.loads(row["payload_json"])["native_text_v1"]["webhook_delivery"]
                    ["deliver_extra"]["trigger"]
                    for row in authority.db._read_all("SELECT request_id, payload_json FROM session_admissions")}
        assert triggers == {D["1"]: "mention", D["2"]: "assigned"}
        assert await _admit(adapter, D["1"], "mention") == "duplicate"
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_a_gateway_started_answer_keeps_one_comment_id_across_retries(monkeypatch):
    """A T3 answer whose create commits but whose response and read-back are lost is retried: the retry
    reuses the comment id, so Linear holds one comment."""
    from types import SimpleNamespace
    from gateway.platforms import webhook_linear
    _runner, authority, adapter, _port, _tiers = await gateway(None, [])
    authority._schedule = lambda ref: None
    monkeypatch.setenv("LINEAR_API_KEY", "lin_test_key")
    posted, lost = [], {"readback": False}

    def linear(query, variables, api_key):
        if "commentCreate" in query:
            posted.append(dict(variables["input"]))
            if len(posted) == 1:
                lost["readback"] = True
                raise ConnectionResetError("connection reset by peer")
            return {"data": {"commentCreate": {"success": True}}}
        if lost["readback"]:
            lost["readback"] = False
            raise ConnectionResetError("connection reset by peer")
        match = next((c for c in posted if c["id"] == variables["id"]), None)
        return {"data": {"comment": match and {"id": match["id"], "body": match["body"],
                                               "parentId": match.get("parentId"), "issue": {"id": match["issueId"]}}}}
    monkeypatch.setattr(webhook_linear, "_linear_graphql", linear)
    try:
        assert await _admit(adapter, D["1"], "mention") == "accepted"
        token = webhook_linear.replying_turn.set(SimpleNamespace(message_id='["plugin","t3-events","k","x"]'))
        try:
            result = await adapter._send_with_retry(f"webhook:linear-comment:thread-{ROOT}", "one T3 answer",
                                                    metadata={"notify": True}, base_delay=0, max_retries=1)
        finally:
            webhook_linear.replying_turn.reset(token)
        assert result.success
        assert len({c["id"] for c in posted}) == 1 and re.fullmatch(webhook_linear._UUID4_RE, posted[0]["id"])
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_a_ledgered_answer_is_redelivered_under_its_own_comment_id(monkeypatch):
    """A final answer whose send failed sits in the delivery ledger; its redelivery posts the same text
    under the same comment id (read back first), never a marked copy or nothing at all."""
    from gateway import delivery_ledger
    from gateway.platforms import webhook_linear
    from gateway.platforms.event import MessageEvent
    from gateway.session_envelope import restore_native
    runner, authority, adapter, _port, _tiers = await gateway(None, [])
    authority._schedule = lambda ref: None
    runner._schedule_flood_redelivery = lambda *a, **kw: None
    monkeypatch.setenv("LINEAR_API_KEY", "lin_test_key")
    attempts, outage = [], {"on": True}

    def linear(query, variables, api_key):
        if "commentCreate" in query:
            attempts.append(dict(variables["input"]))
        if outage["on"]:
            raise TimeoutError("timed out")
        return {"data": {"comment": None, "commentCreate": {"success": True}}}
    monkeypatch.setattr(webhook_linear, "_linear_graphql", linear)
    try:
        await _admit(adapter, D["1"], "mention")
        (row,) = authority.db._read_all("SELECT payload_json FROM session_admissions")
        source = restore_native(json.loads(row["payload_json"]), runner).source
        session_key = runner._session_key_for_source(source)
        turn_id = '["plugin","t3-events","k","event-one"]'
        event = MessageEvent(text="[T3 STATE EVENT] ready", source=source, message_id=turn_id, internal=True)
        result, _ = await adapter.send_final_ledgered(event, session_key, "T3 final answer", {"notify": True},
                                                      reply_to=None)
        assert not result.success and len(attempts) == 1
        oid = delivery_ledger.compute_obligation_id(session_key, turn_id, "T3 final answer")
        # What the ledger kept for this answer (a live owner's row: the runtime/boot sweeps claim it the same way).
        conn = delivery_ledger._connect()
        try:
            (stored_ref,) = conn.execute("SELECT reply_ref FROM delivery_obligations WHERE obligation_id=?",
                                         (oid,)).fetchone()
        finally:
            conn.close()
        assert stored_ref == attempts[0]["id"], "the obligation records the id the answer was attempted under"
        claimed = [delivery_ledger._claimed_row(oid, session_key, "webhook", source.chat_id, None, "T3 final answer",
                                                1, "default", needs_marker=True, reply_ref=stored_ref)]
        outage["on"] = False
        assert await runner._redeliver_claimed_obligations(claimed) == 1
        assert len(attempts) == 2 and attempts[1]["id"] == attempts[0]["id"], attempts
        assert attempts[1]["body"] == "T3 final answer", "a keyed answer is resent unmarked"
    finally:
        await adapter.disconnect()


async def gateway(peer, posted):
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.webhook import WebhookAdapter
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    runner = GatewayRunner()
    runner._running = True  # a started gateway: plugin injection (t3-events) accepts only then
    authority = await initialize_session_authority(runner, profile_id="default", instance_id=f"lane-{len(posted)}")
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": 0, "secret": _SECRET.decode(), "routes": {"linear-comment": dict(ROUTE)}}))
    adapter.gateway_runner = runner
    runner.adapters[Platform.WEBHOOK] = adapter
    runner._wire_adapter_handlers(adapter)
    tiers = []
    real = adapter.toolsets_for_source
    adapter.toolsets_for_source = lambda source: tiers.append((source.chat_id, real(source))) or tiers[-1][1]
    assert await adapter.connect()
    port = next(iter(adapter._runner.sites))._server.sockets[0].getsockname()[1]
    return runner, authority, adapter, port, tiers


async def probe(peer):
    import aiohttp
    from gateway.platforms import webhook_linear
    from gateway.platforms.webhook import _WebhookDeliveryIdentity
    from gateway.run_runtime import recover_profile_native_sessions
    from tests.waiting import async_wait_until

    posted = []

    def linear(query, variables, api_key):
        assert api_key == "lin_test_key"
        if "commentCreate" in query:
            posted.append(dict(variables["input"]))
            return {"data": {"commentCreate": {"success": True, "comment": {"id": variables["input"]["id"]}}}}
        match = next((c for c in posted if c["id"] == variables["id"]), None)
        return {"data": {"comment": None if match is None else {
            "id": match["id"], "body": match["body"], "parentId": match.get("parentId"),
            "issue": {"id": match["issueId"]}}}}

    webhook_linear._linear_graphql = linear
    runner, authority, adapter, port, tiers = await gateway(peer, posted)

    def rows(request_id):
        return authority.db._read_all(
            "SELECT target_session_id, status FROM session_admissions WHERE request_id=?", (request_id,))

    async def post(client, delivery_id, *, text, author="member", thread=ROOT, issue=ISSUE, trigger="mention",
                   expect=202, port=port, **extra):
        body = json.dumps({"text": text, "author": author, "thread": thread, "issue": issue, "parent": thread,
                           "trigger": trigger, **extra}).encode()
        response = await client.post(f"http://127.0.0.1:{port}/webhooks/linear-comment", data=body, headers={
            "X-Request-ID": delivery_id,
            "X-Hub-Signature-256": "sha256=" + hmac.new(_SECRET, body, hashlib.sha256).hexdigest()})
        assert response.status == expect, (response.status, await response.text())

    async def settled(adapter):
        await async_wait_until(lambda: not adapter._background_tasks, timeout=40, what="deliveries to finalize")

    async with aiohttp.ClientSession() as client:
        # A gate that cannot read canonical context: retryable 503, nothing admitted.
        await post(client, D["1"], text="x", retry=True, expect=503)
        assert rows(D["1"]) == []

        # Staff comments in one thread: one session, one turn at a time, each answer keyed on its delivery.
        await post(client, D["1"], text="BLOCK_FIFO TAG-one", author="staff")
        assert await asyncio.to_thread(peer.blocked.wait, 15)
        await post(client, D["2"], text="TAG-two", author="staff")
        (first,), (second,) = rows(D["1"]), rows(D["2"])
        assert first["target_session_id"] == second["target_session_id"]
        assert (first["status"], second["status"]) == ("started", "queued"), "a follow-up waits for the running turn"
        await post(client, D["1"], text="BLOCK_FIFO TAG-one", author="staff", expect=200)  # provider retry
        peer.blocked.clear()
        peer.release.set()
        await settled(adapter)
        peer.release.clear()
        thread_order = [c["id"] for c in posted]
        assert thread_order == [D["1"], D["2"]], posted
        assert [c["body"] for c in posted] == ["ANSWER TAG-one", "ANSWER TAG-two"]
        assert all(c["issueId"] == ISSUE and c["parentId"] == ROOT for c in posted)
        thread_sid = first["target_session_id"]
        assert authority.db.get_session(thread_sid)["ended_at"] is None, "a thread session outlives its turns"
        thread_chat = f"webhook:linear-comment:thread-{ROOT}"
        assert {tuple(t) for c, t in tiers if c == thread_chat} == {("memory", "todo")} or \
            {tuple(sorted(t)) for c, t in tiers if c == thread_chat} == {("memory", "todo")}

        # A member stays one-shot with the read-only toolsets, even when the body names a tier and a thread.
        await post(client, D["3"], text="TAG-three", issue=OTHER_ISSUE,
                   __hermes_toolset_tier__="staff", __hermes_session_key__=ROOT)
        await settled(adapter)
        one_shot = _WebhookDeliveryIdentity.from_parts(None, "linear-comment", D["3"]).session_chat_id
        (third,) = rows(D["3"])
        assert third["target_session_id"] != thread_sid
        assert [t for c, t in tiers if c == one_shot] and all(t == ["todo"] for c, t in tiers if c == one_shot)
        assert posted[-1]["id"] == D["3"] and posted[-1]["issueId"] == OTHER_ISSUE
        assert authority.db.get_session(third["target_session_id"])["ended_at"] is not None

        # Silence: a mention is owed an answer; an assignment-only wake is not.
        await post(client, D["4"], text="SAY_NOTHING")
        await post(client, D["5"], text="SAY_NOTHING", trigger="assigned")
        await settled(adapter)
        assert [(c["id"], c["body"]) for c in posted if c["id"] in (D["4"], D["5"])] == [
            (D["4"], webhook_linear.LINEAR_SILENCE_ACK)]

        # A gateway-started turn (a T3 state event) answers in the thread under a reply id of its own.
        # The exact call t3_events.deliver_pending makes (hermes/t3ctl bin/t3-events in the monorepo).
        entry = runner.session_store.lookup_by_session_key(authority.sessions[thread_sid].route)
        assert not await runner._dispatch_plugin_message_injection(
            session_key=entry.session_key, content="stale", plugin_id="t3-events", expected_session_id="reset-since")
        assert await runner._dispatch_plugin_message_injection(
            session_key=entry.session_key, content="[T3 STATE EVENT] TAG-tee", plugin_id="t3-events",
            expected_session_id=entry.session_id)
        await async_wait_until(lambda: any(c["body"] == "ANSWER TAG-tee" for c in posted), timeout=40,
                               what="the T3 turn's answer")
        tee = next(c for c in posted if c["body"] == "ANSWER TAG-tee")
        assert re.fullmatch(webhook_linear._UUID4_RE, tee["id"]) and tee["id"] not in D.values()
        assert tee["issueId"] == ISSUE and tee["parentId"] == ROOT

        # Restart: a delivery committed but never run is answered by the next gateway, keyed on itself.
        authority._schedule = lambda ref: None
        await post(client, D["6"], text="TAG-six", author="staff")
        assert rows(D["6"])[0]["status"] == "queued"
    await adapter.disconnect()
    runner2, authority2, adapter2, _port, tiers2 = await gateway(peer, posted)
    await recover_profile_native_sessions(runner2, authority2)
    await async_wait_until(lambda: any(c["body"] == "ANSWER TAG-six" for c in posted), timeout=40,
                           what="the recovered delivery's answer")
    six = next(c for c in posted if c["body"] == "ANSWER TAG-six")
    assert six["issueId"] == ISSUE and six["parentId"] == ROOT
    assert [t for c, t in tiers2 if c == thread_chat] and all(sorted(t) == ["memory", "todo"] for c, t in tiers2 if c == thread_chat)
    await adapter2.disconnect()
    Path(os.environ["HERMES_HOME"], "lane-receipt.json").write_text(json.dumps({
        "thread_order": thread_order, "restart_reply": six["id"]}))


def main():
    from http.server import ThreadingHTTPServer
    import threading
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(1, str(Path(__file__).parent / "fixtures"))
    from kosmi_peer import ScriptedPeer

    peer = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedPeer)
    peer.requests, peer.metadata_requests = [], []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{peer.server_port}/v1"
    home = Path(os.environ["HERMES_HOME"])
    os.environ.update(OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=url, LINEAR_API_KEY="lin_test_key")
    (home / "scripts").mkdir(exist_ok=True)
    (home / "scripts" / "gate.py").write_text(GATE)
    (home / "config.yaml").write_text(
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
