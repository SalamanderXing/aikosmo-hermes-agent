"""The startup API media pass never runs on the gateway loop, and its transcript scan stays cheap.

On 2026-11-04 the orphan-image check read Kosmi's whole transcript (297k rows, 847 MB) once per
candidate digest on the loop at every boot: 60-80 s with no heartbeat, so Discord dropped right
after connecting."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session_api_turn import admit_api_turn
from gateway.session_ingress_media import _history_mentions, _media_root, collect_unheld_api_images
from gateway.session_runtime_workers import join_authority_work, mutation_tasks


def _plant(*digests, name=None):
    root = _media_root()
    planted = []
    for digest in digests:
        directory = root / digest
        directory.mkdir(parents=True)
        path = directory / (name or f'api_{digest[:32]}.png')
        path.write_bytes(b'img' + digest.encode())
        planted.append((digest, path))
    return planted


def _note(owner, session_id, content):
    if owner.db.get_session(session_id) is None:
        owner.db.create_session(session_id, source='cli')
    owner.db.append_message(session_id, 'user', content)


async def _settled(owner):
    await asyncio.gather(*list(getattr(owner, '_mutation_tasks', ())), return_exceptions=True)


def _blocked_collect(monkeypatch):
    release, swept = threading.Event(), []

    def slow_collect(db):
        swept.append(threading.get_ident())
        assert release.wait(5)

    monkeypatch.setattr('gateway.session_ingress_media.collect_unheld_api_images', slow_collect)
    return release, swept


async def _until(predicate):
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError('condition never held')


def _host():
    host = object.__new__(GatewayRunner)
    host.adapters = {}
    host._sync_voice_mode_state_to_adapter = host._bind_voice_input_callback = lambda adapter: None
    return host


def test_prefiltered_scan_still_finds_every_spelling(api, owner):
    """Raw and JSON-escaped mentions with either separator keep their image; a row naming only the
    digest directory, never the file, holds nothing."""
    (raw, raw_path), (escaped, escaped_path), (back, back_path), (orphan, orphan_path), (decoy, decoy_path) = _plant(
        *(f'{index:064x}' for index in range(1, 6)))
    _note(owner, 'keeper', f'saved at {raw}/{raw_path.name}')
    _note(owner, 'keeper', json.dumps({'note': f'{escaped}/{escaped_path.name}'}))
    _note(owner, 'keeper', json.dumps({'note': f'C:\\cache\\{back}\\{back_path.name}'}))
    _note(owner, 'keeper', f'the directory {decoy} alone')
    collect_unheld_api_images(owner.db)
    assert raw_path.exists() and escaped_path.exists() and back_path.exists()
    assert not orphan_path.exists() and not decoy_path.exists()


def test_scan_without_a_common_name_prefix_has_no_prefilter(owner):
    """Candidates whose names share no prefix are still matched (the query binds no prefix)."""
    _note(owner, 'keeper', 'first ' + 'a' * 64 + '/alpha.png')
    _note(owner, 'keeper', 'second ' + 'b' * 64 + '/beta.png')
    references = [{'path': f'/m/{"a" * 64}/alpha.png', 'sha256': 'a' * 64},
                  {'path': f'/m/{"b" * 64}/beta.png', 'sha256': 'b' * 64},
                  {'path': f'/m/{"c" * 64}/gamma.png', 'sha256': 'c' * 64}]
    with owner.db._read_ctx() as conn:
        assert _history_mentions(conn, references) == {references[0]['path'], references[1]['path']}
        assert _history_mentions(conn, references, after_id=1) == {references[1]['path']}


@pytest.mark.asyncio
async def test_publishing_the_api_adapter_sweeps_media_off_the_loop(api, owner, monkeypatch):
    """Queued API turns recover at publication; the media pass runs in a worker thread while the
    loop keeps serving."""
    admit_api_turn(api, session_id='queued', request_id='queued-turn',
                   user_message='pending', conversation_history=[])
    scheduled = []
    monkeypatch.setattr(owner, '_schedule', lambda ref: scheduled.append(ref.session_id))
    release, swept = _blocked_collect(monkeypatch)
    host = _host()
    host._publish_primary_adapter(Platform.API_SERVER, api)
    assert host.adapters[Platform.API_SERVER] is api
    assert scheduled == ['queued']  # recovery never waits for the media pass
    await _until(lambda: swept)  # the loop runs while the worker holds the pass
    release.set()
    await _settled(owner)
    assert swept == [swept[0]] and swept[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_retirement_and_shutdown_count_the_sweep_until_its_thread_returns(api, owner, monkeypatch):
    """The pass is the store's owner work: retirement cannot release ownership and shutdown cannot
    close the store while its thread may still write, even after its task was cancelled."""
    release, swept = _blocked_collect(monkeypatch)
    _host()._publish_primary_adapter(Platform.API_SERVER, api)
    await _until(lambda: swept)
    (task,) = mutation_tasks(owner)
    with pytest.raises(TimeoutError):
        await join_authority_work(owner, 0.1)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done() and mutation_tasks(owner) == [task]  # still counted: the thread is live
    release.set()
    await join_authority_work(owner, 5)
    assert task.cancelled() and mutation_tasks(owner) == []


@pytest.mark.asyncio
async def test_the_background_sweep_still_collects_orphans(api, owner):
    (kept, kept_path), (_orphan, orphan_path) = _plant('d' * 64, 'e' * 64)
    _note(owner, 'keeper', f'{kept}/{kept_path.name}')
    _host()._publish_primary_adapter(Platform.API_SERVER, api)
    await _settled(owner)
    assert kept_path.exists() and not orphan_path.exists()


@pytest.mark.asyncio
async def test_a_failed_background_sweep_is_logged_and_leaves_the_gateway_up(api, owner, monkeypatch, caplog):
    def broken(db):
        raise RuntimeError('compaction blew up')

    monkeypatch.setattr('gateway.session_api_media.compact_settled_api_payloads', broken)
    host = _host()
    host._publish_primary_adapter(Platform.API_SERVER, api)
    await _settled(owner)
    assert host.adapters[Platform.API_SERVER] is api
    assert 'Startup API media sweep failed' in caplog.text


def test_without_a_running_loop_the_sweep_runs_inline(api, owner):
    (_orphan, orphan_path), = _plant('f' * 64)
    _host()._publish_primary_adapter(Platform.API_SERVER, api)
    assert not orphan_path.exists()


@pytest.mark.asyncio
async def test_a_profile_served_later_sweeps_in_a_worker_then_recovers(api, owner, monkeypatch):
    from gateway import run_runtime
    order = []
    monkeypatch.setattr('gateway.session_ingress_media.collect_unheld_api_images',
                        lambda db: order.append(('sweep', threading.get_ident())))
    monkeypatch.setattr('gateway.session_api_turn.recover_authority_api_turns',
                        lambda adapter, authority, sweep_media=True: order.append(('recover', sweep_media)))
    runner = SimpleNamespace(adapters={Platform.API_SERVER: api})
    await run_runtime._recover_published_api_turns(runner, owner)
    assert [step for step, _ in order] == ['sweep', 'recover']
    assert order[0][1] != threading.get_ident() and order[1][1] is False


@pytest.mark.asyncio
async def test_a_profile_served_later_counts_its_sweep_as_owner_work(api, owner, monkeypatch):
    from gateway import run_runtime
    release, swept = _blocked_collect(monkeypatch)
    monkeypatch.setattr('gateway.session_api_turn.recover_authority_api_turns',
                        lambda adapter, authority, sweep_media=True: None)
    serve = asyncio.ensure_future(run_runtime._recover_published_api_turns(
        SimpleNamespace(adapters={Platform.API_SERVER: api}), owner))
    await _until(lambda: swept)
    assert len(mutation_tasks(owner)) == 1
    serve.cancel()  # a cancelled serve leaves the pass counted for its retirement
    await asyncio.sleep(0.05)
    assert len(mutation_tasks(owner)) == 1
    release.set()
    await join_authority_work(owner, 5)


@pytest.mark.asyncio
async def test_a_worker_that_fails_after_a_cancel_still_reports_the_cancellation(owner, caplog):
    from gateway.session_runtime_workers import track_worker_thread
    release = threading.Event()

    def work():
        assert release.wait(5)
        raise ValueError('late failure')

    task = track_worker_thread(owner, work)
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()
    release.set()
    await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled() and mutation_tasks(owner) == []
    assert 'failed after its task was cancelled' in caplog.text


@pytest.mark.asyncio
async def test_a_worker_failure_without_a_cancel_is_the_task_result(owner):
    from gateway.session_runtime_workers import track_worker_thread

    def work():
        raise ValueError('plain failure')

    task = track_worker_thread(owner, work)
    with pytest.raises(ValueError, match='plain failure'):
        await task
