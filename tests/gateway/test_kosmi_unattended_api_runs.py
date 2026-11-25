"""AI Kosmo: a /v1/runs turn its caller marks ``unattended`` settles when its execution is lost.

A backend job (a ticket analysis) polls its run and fails it on its own; nobody types /discard
for it. Upstream treats every API turn as a person's, so a turn lost in a gateway restart stayed
``unknown`` and paused its session for good (ticket 4994, 2026-11-24). With the marker it settles
``interrupted`` at boot like a webhook or Cloud turn, the next turn runs, and the run poll says why.
A turn without the marker still pauses. Real SQLite store, two real owner epochs, real HTTP doors.
"""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig
from gateway.platforms.api_server_authority_runs import LOST_TURN_ERROR
from gateway.session import SessionStore
from gateway.session_api import restore_api_session
from gateway.session_authority import admission_row, initialize_session_authority
from gateway.session_unattended_unknown import is_unattended
from tests.gateway.test_api_unknown_resolution import _AUTH_HEADERS, _adapter, _close_fixture, _runs_client


async def _lost_head(tmp_path, monkeypatch, *, unattended):
    """Owner 1 admits two runs over HTTP (head + follower), claims the head and dies; owner 2
    boots on the same store. Returns the fixture tuple of test_api_unknown_resolution."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    executed = []

    async def answer(event):
        from gateway.session_results import execution_result
        executed.append(event.text)
        execution_result.get()["result"] = {"final_response": "ACK_" + event.text, "messages": []}
        return "ACK_" + event.text

    from persistence_assembly import open_persistence, profile_scope
    runner = SimpleNamespace(session_store=store, _draining=False, _handle_message=answer,
                             persistence=open_persistence(profile_scope(home=tmp_path)))
    db = runner.persistence
    first = await initialize_session_authority(runner, profile_id="default", instance_id="owner-1")
    adapter = _adapter(runner, db)
    runner._adapter_for_source = lambda source: adapter
    first._schedule = lambda ref: None
    client = _runs_client(adapter)
    await client.start_server()
    body = {"session_id": "ticket-session", **({"unattended": True} if unattended else {})}
    run_ids = []
    for text in ("HEAD", "FOLLOWER"):
        response = await client.post("/v1/runs", json=body | {"input": text}, headers=_AUTH_HEADERS)
        assert response.status == 202, await response.text()
        run_ids.append((await response.json())["run_id"])
    tasks = list(adapter._active_run_tasks.values())
    await asyncio.sleep(0)
    db.admissions.claim(epoch=first.epoch, session_id="ticket-session")
    await client.close()

    owner = await initialize_session_authority(runner, profile_id="default", instance_id="owner-2")
    restore_api_session(owner, "ticket-session")
    fresh = _adapter(runner, db)
    runner._adapter_for_source = lambda source: fresh
    client = _runs_client(fresh)
    await client.start_server()
    return client, (adapter, fresh), store, owner, run_ids, executed, tasks


def _statuses(owner):
    return [(row["status"], row["outcome"]) for row in map(admission_row, owner.persistence.admissions.list(
        session_id="ticket-session", pending_only=False))]


@pytest.mark.asyncio
async def test_unattended_api_run_lost_in_restart_settles_and_the_session_moves_on(tmp_path, monkeypatch):
    client, adapters, store, owner, (head, follower), executed, tasks = await _lost_head(
        tmp_path, monkeypatch, unattended=True)
    try:
        rows = [admission_row(r) for r in owner.persistence.admissions.list(session_id="ticket-session",
                                                                            pending_only=False)]
        assert rows[0]["payload"]["api_turn_v1"]["unattended"] is True
        assert _statuses(owner) == [("terminal", "interrupted"), ("queued", None)]

        response = await client.get("/v1/runs/" + head, headers=_AUTH_HEADERS)
        assert response.status == 200, await response.text()
        projected = await response.json()
        assert projected["status"] == "cancelled" and projected["error"] == LOST_TURN_ERROR

        # The follower is not stuck behind the lost head: the next drain runs it, once.
        owner._schedule(SimpleNamespace(profile_id="default", session_id="ticket-session"))
        await owner.sessions["ticket-session"].task
        assert executed == ["FOLLOWER"]
        assert _statuses(owner) == [("terminal", "interrupted"), ("terminal", "completed")]
        done = await (await client.get("/v1/runs/" + follower, headers=_AUTH_HEADERS)).json()
        assert done["status"] == "completed" and "error" not in done
    finally:
        await _close_fixture(client, adapters, store, tasks)


@pytest.mark.asyncio
async def test_api_run_without_the_marker_still_pauses_for_a_person(tmp_path, monkeypatch):
    client, adapters, store, owner, (head, _follower), executed, tasks = await _lost_head(
        tmp_path, monkeypatch, unattended=False)
    try:
        assert _statuses(owner) == [("unknown", None), ("queued", None)]
        projected = await (await client.get("/v1/runs/" + head, headers=_AUTH_HEADERS)).json()
        assert projected["status"] == "unknown" and "error" not in projected
        assert executed == []
    finally:
        await _close_fixture(client, adapters, store, tasks)


@pytest.mark.asyncio
async def test_unattended_must_be_a_boolean(tmp_path, monkeypatch):
    client, adapters, store, owner, _runs, _executed, tasks = await _lost_head(
        tmp_path, monkeypatch, unattended=False)
    try:
        response = await client.post("/v1/runs", json={"input": "x", "session_id": "other", "unattended": "yes"},
                                     headers=_AUTH_HEADERS)
        assert response.status == 400
        assert (await response.json())["error"]["code"] == "invalid_params"
    finally:
        await _close_fixture(client, adapters, store, tasks)


def test_is_unattended_reads_only_the_committed_api_marker():
    def row(api_turn):
        return {"principal_id": "api", "payload": {"text": "x", "api_turn_v1": api_turn}}
    assert is_unattended(row({"history": None, "settings": {}, "unattended": True}))
    assert not is_unattended(row({"history": None, "settings": {}}))
    assert not is_unattended(row({"history": None, "settings": {}, "unattended": "true"}))
