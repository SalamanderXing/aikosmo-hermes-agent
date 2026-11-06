"""AI Kosmo: a keyed plugin turn (the t3-events outbox reporting a finished T3 thread) is admitted
into a live Discord thread whose stored origin has drifted from the last committed human message:
auto-thread markers, a renamed thread title, another member speaking last. ``snapshot_automation``
refused that origin with ``admission_conflict`` on every attempt, so Kosmi never reported the
completion. Real GatewayRunner, DiscordAdapter, SessionAuthority and drain over SQLite; the turn is a
stand-in."""
import asyncio
import dataclasses

import pytest

from tests.waiting import async_wait_until


async def _thread_gateway(monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42,43')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id='sent')
    monkeypatch.setattr(adapter, 'send', send)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    runner._running = True  # plugin injection only dispatches into a running gateway
    executed = []

    async def turn(event, source, quick_key, run_generation):
        executed.append((event.text, event.internal))
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'reply', 'messages': []}
        return 'reply'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    return runner, adapter, authority, executed


def _thread_source(adapter, *, user_id, user_name, chat_name, **extra):
    return adapter.build_source(
        chat_id='900', chat_name=chat_name, chat_type='thread', user_id=user_id, user_name=user_name,
        thread_id='900', parent_chat_id='800', scope_id='700', **extra)


async def _human(adapter, executed, source, text, message_id):
    from gateway.platforms.event import MessageEvent
    await adapter.handle_message(MessageEvent(text=text, source=source, message_id=message_id))
    await async_wait_until(lambda: (text, False) in executed, timeout=15, what=f'human turn {text!r}')


@pytest.mark.asyncio
async def test_t3_completion_reaches_a_drifted_auto_thread(monkeypatch):
    runner, adapter, authority, executed = await _thread_gateway(monkeypatch)
    original = 'Guild / #daniele-kosmi / when emails come from booking.com...'
    first = _thread_source(adapter, user_id='42', user_name='Daniele', chat_name=original)
    await _human(adapter, executed, first, 'A', 'A')
    # Another member speaks last in the same thread: the committed envelope is now theirs.
    await _human(adapter, executed, _thread_source(adapter, user_id='43', user_name='Giulio', chat_name=original),
                 'B', 'B')

    key = runner.session_store._generate_session_key(first)
    entry = runner.session_store.lookup_by_session_key(key)
    # The stored origin as a live Kosmi auto-thread carries it: auto-thread markers and the title
    # the thread was renamed to after the first reply.
    entry.origin = dataclasses.replace(
        entry.origin, auto_thread_created=True, auto_thread_initial_name=original,
        chat_name='Guild / #daniele-kosmi / Booking.com domain block')

    from gateway.session_automation import snapshot_automation
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session_identity import replace_source
    from hermes_state_runtime import RuntimeStoreError
    stale = MessageEvent(text='x', message_type=MessageType.TEXT, internal=True, message_id='probe',
                         source=replace_source(runner._restored_source(entry)),
                         metadata={'gateway_session_key': key, 'gateway_session_id': entry.session_id})
    with pytest.raises(RuntimeStoreError) as refused:
        snapshot_automation(authority, adapter, stale, 'probe')
    assert refused.value.reason == 'admission_conflict'  # the stored origin alone is refused

    assert await runner._dispatch_plugin_message_injection(
        session_key=key, content='[T3 STATE EVENT] completed', plugin_id='t3-events',
        expected_session_id=entry.session_id) is True
    await async_wait_until(lambda: ('[T3 STATE EVENT] completed', True) in executed, timeout=15,
                           what='the T3 completion turn')
    assert executed.count(('[T3 STATE EVENT] completed', True)) == 1


@pytest.mark.asyncio
async def test_keyed_injection_without_a_committed_human_turn_is_refused(monkeypatch):
    runner, adapter, authority, executed = await _thread_gateway(monkeypatch)
    from gateway.session import SessionSource
    source = _thread_source(adapter, user_id='42', user_name='Daniele', chat_name='Guild / #c / t')
    key = runner.session_store._generate_session_key(source)
    entry = runner.session_store.get_or_create_session(source)
    assert isinstance(entry.origin, SessionSource)
    assert await runner._dispatch_plugin_message_injection(
        session_key=key, content='orphan', plugin_id='t3-events') is False
    await asyncio.sleep(0.2)
    assert ('orphan', True) not in executed


@pytest.mark.asyncio
async def test_t3_outbox_event_is_delivered_into_a_drifted_auto_thread(monkeypatch, tmp_path):
    t3_events = pytest.importorskip('t3_events')  # the image copies t3ctl's bin/t3-events here
    runner, adapter, authority, executed = await _thread_gateway(monkeypatch)
    original = 'Guild / #daniele-kosmi / who developed the booking widget...'
    first = _thread_source(adapter, user_id='42', user_name='Daniele', chat_name=original)
    await _human(adapter, executed, first, 'A', 'A')
    key = runner.session_store._generate_session_key(first)
    entry = runner.session_store.lookup_by_session_key(key)
    entry.origin = dataclasses.replace(entry.origin, auto_thread_created=True, auto_thread_initial_name=original,
                                       chat_name='Guild / #daniele-kosmi / Ski email layout')

    path = tmp_path / 'events.sqlite'
    receipt = {'threadId': 't3-thread', 'dispatchId': 'aik-3946-1', 'messageId': 'm',
               'previousTurnId': 'old', 'turnId': 'new'}
    assert t3_events.register(receipt, {
        'T3CTL_EVENTS_ENABLED': '1', 'HERMES_SESSION_ID': entry.session_id,
        'HERMES_SESSION_KEY': key, 'HERMES_SESSION_PLATFORM': 'discord'}, path)
    with t3_events.connect(path) as db:
        subscription = dict(db.execute('SELECT * FROM subscriptions').fetchone())
    t3_events.observe(subscription, {'reason': 'settled', 'turn': 'completed', 'turnId': 'new',
                                    'lastAssistant': 'PR opened, checks running'}, path)
    await t3_events.deliver_pending(runner, path)
    assert t3_events.pending(path) == []
    with t3_events.connect(path) as db:
        assert [r['disposition'] for r in db.execute('SELECT disposition FROM events')] == ['delivered']
    await async_wait_until(lambda: any(internal and 'PR opened, checks running' in text
                                       for text, internal in executed), timeout=15, what='the T3 completion turn')
