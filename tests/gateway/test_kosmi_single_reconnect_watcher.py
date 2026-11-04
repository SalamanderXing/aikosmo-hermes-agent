"""AI Kosmo: one reconnect watcher, and a reconnect that loses a race never replaces the live adapter.

Production (2026-11-03, 2026-11-04): a startup step blocked the event loop for 60-80 s, Discord's
heartbeat starved and its socket closed right after "✓ discord connected". The fatal handler spawned a
reconnect watcher through ``_ensure_reconnect_watcher_running`` (startup had not spawned one yet), then
startup spawned a second. Both reconnected Discord: the first installed adapter A, the second published
adapter B, hit ``KeyError`` popping the already-popped queue entry and disposed B. ``self.adapters`` kept
the disposed B while A kept receiving messages, each refused ``RuntimeStoreError('not_found')``, and with
the platform no longer queued nothing ever retried it.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from tests.gateway.test_platform_reconnect import (
    StubAdapter, TestReconnectKeepsInboundDedup as _ReconnectFixture, _make_runner,
)


def _runner_recording_spawns(spawned):
    runner = _make_runner()
    runner._background_tasks = set()

    def _fake_spawn(coro_factory, name, **kwargs):
        spawned.append(name)
        handle = MagicMock()
        handle.done.return_value = False
        on_spawn = kwargs.get("on_spawn")
        if on_spawn is not None:
            on_spawn(handle)
        return handle

    runner._spawn_supervised = _fake_spawn
    runner._scale_to_zero_should_arm = lambda: False
    runner._log_scale_to_zero_not_armed_reason = lambda: None
    return runner


class TestStartupKeepsTheWatcherAFatalAlreadySpawned:
    def test_startup_does_not_spawn_a_second_reconnect_watcher(self):
        spawned = []
        runner = _runner_recording_spawns(spawned)
        # A fatal during startup: queued, and the ensure hook spawned the watcher.
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": runner.config.platforms[Platform.TELEGRAM], "attempts": 0, "next_retry": 0,
        }
        runner._ensure_reconnect_watcher_running()
        assert spawned.count("platform_reconnect_watcher") == 1

        runner._start_spawn_background_watchers()

        assert spawned.count("platform_reconnect_watcher") == 1

    def test_startup_spawns_the_watcher_when_none_runs(self):
        spawned = []
        runner = _runner_recording_spawns(spawned)

        runner._start_spawn_background_watchers()

        assert spawned.count("platform_reconnect_watcher") == 1

    def test_startup_replaces_a_dead_watcher(self):
        spawned = []
        runner = _runner_recording_spawns(spawned)
        dead = MagicMock()
        dead.done.return_value = True
        runner._reconnect_watcher_task = dead

        runner._start_spawn_background_watchers()

        assert spawned.count("platform_reconnect_watcher") == 1
        assert runner._reconnect_watcher_task is not dead


class TestSupersededReconnectKeepsTheLiveAdapter:
    @pytest.mark.asyncio
    async def test_a_reconnect_that_lost_the_race_is_disposed_not_published(self):
        runner = _make_runner()
        runner._update_platform_runtime_status = MagicMock()
        runner._wire_adapter_handlers = MagicMock()
        runner._install_reconnected_adapter = AsyncMock()
        platform = Platform.TELEGRAM
        runner._failed_platforms[platform] = {
            "config": PlatformConfig(enabled=True, token="test"), "attempts": 0, "next_retry": 0,
        }
        winner = StubAdapter()
        loser = StubAdapter()

        async def connect_while_the_other_reconnect_installs(*, is_reconnect=False):
            # The concurrent reconnect finishes first: it publishes its adapter and pops the queue entry.
            runner.adapters[platform] = winner
            del runner._failed_platforms[platform]
            return True

        loser.connect = connect_while_the_other_reconnect_installs
        disposed = []

        async def _dispose(adapter):
            disposed.append(adapter)

        with patch.object(runner, "_create_adapter", return_value=loser), \
                patch("gateway.run._dispose_unused_adapter", side_effect=_dispose):
            await runner._reconnect_failed_platform(platform, time.monotonic())

        runner._install_reconnected_adapter.assert_not_awaited()
        assert runner.adapters[platform] is winner
        assert disposed == [loser]
        assert platform not in runner._failed_platforms

    @pytest.mark.asyncio
    async def test_the_only_reconnect_still_installs(self):
        runner = _make_runner()
        runner._update_platform_runtime_status = MagicMock()
        runner._wire_adapter_handlers = MagicMock()
        runner._install_reconnected_adapter = AsyncMock()
        platform = Platform.TELEGRAM
        runner._failed_platforms[platform] = {
            "config": PlatformConfig(enabled=True, token="test"), "attempts": 0, "next_retry": 0,
        }
        adapter = StubAdapter()

        with patch.object(runner, "_create_adapter", return_value=adapter):
            await runner._reconnect_failed_platform(platform, time.monotonic())

        runner._install_reconnected_adapter.assert_awaited_once_with(platform, adapter)


class TestSupervisedRespawnDoesNotAddAWatcher:
    @pytest.mark.asyncio
    async def test_a_fatal_during_supervised_backoff_leaves_one_watcher(self, monkeypatch):
        """The watcher crashed and its supervised respawn is sleeping; a fatal meanwhile makes
        _ensure_reconnect_watcher_running start a replacement, then the respawn starts another."""
        runner = _make_runner()
        runner._background_tasks = set()
        runner._reconnect_watcher_task = None
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": runner.config.platforms[Platform.TELEGRAM], "attempts": 0,
            "next_retry": time.monotonic() + 3600,
        }
        callers = []

        async def _record(platform, now):
            callers.append(asyncio.current_task())

        runner._reconnect_failed_platform = _record
        backoff_entered, release_backoff = asyncio.Event(), asyncio.Event()
        real_sleep = asyncio.sleep

        async def _sleep(delay):
            if delay == 919:
                backoff_entered.set()
                await release_backoff.wait()
            else:
                await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", _sleep)
        runner._supervised_backoff = lambda _attempt: 919
        crashed = []

        async def _watcher():
            if not crashed:
                crashed.append(True)
                raise RuntimeError("first watcher crashed")
            await GatewayRunner._platform_reconnect_watcher(runner)

        runner._platform_reconnect_watcher = _watcher

        async def _spin(n=200):
            for _ in range(n):
                await real_sleep(0)

        try:
            runner._spawn_reconnect_watcher()
            await asyncio.wait_for(backoff_entered.wait(), 5)
            runner._ensure_reconnect_watcher_running()
            replacement = runner._reconnect_watcher_task
            await _spin()
            assert callers and set(callers) == {replacement}
            release_backoff.set()
            await _spin()
            respawned = runner._reconnect_watcher_task
            assert respawned is not replacement
            callers.clear()
            await _spin()

            assert replacement.done() and replacement.exception() is None
            assert callers and set(callers) == {respawned}
        finally:
            runner._running = False
            tasks = list(runner._background_tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class TestSupersededReconnectKeepsHeldInbound:
    @pytest.mark.asyncio
    async def test_held_inbound_goes_to_the_requeued_predecessor_when_the_winner_failed_again(self):
        runner = _make_runner()
        runner._update_platform_runtime_status = MagicMock()
        runner._wire_adapter_handlers = MagicMock()
        runner._sync_voice_mode_state_to_adapter = MagicMock()
        runner._bind_voice_input_callback = MagicMock()
        runner._schedule_planned_restart_replay = MagicMock()
        runner._schedule_resume_pending_sessions = MagicMock()
        runner._redeliver_failed_obligations_for_platform = AsyncMock()
        runner._run_in_executor_with_context = AsyncMock()
        runner._ensure_reconnect_watcher_running = MagicMock()
        telegram = _ReconnectFixture._telegram_cls()
        platform = Platform.TELEGRAM
        winner, loser, replacement = [telegram(True) for _ in range(3)]
        held = _ReconnectFixture._event("acked by the losing connection")
        original_info = {"config": runner.config.platforms[platform], "attempts": 0, "next_retry": 0}
        runner._failed_platforms[platform] = original_info
        winner._set_fatal_error("network", "winner failed again", retryable=True)

        async def connect(*, is_reconnect=False):
            # Another reconnect won (popped original_info), then its live adapter failed again.
            runner._failed_platforms.pop(platform)
            runner._queue_retryable_fatal_platform(winner)
            loser._held_inbound_events = [held]
            return True

        loser.connect = connect
        with patch.object(runner, "_create_adapter", side_effect=[loser, replacement]), \
                patch("gateway.channel_directory.build_channel_directory", new=AsyncMock()):
            await runner._reconnect_failed_platform(platform, time.monotonic())
            assert runner._failed_platforms[platform] is not original_info
            await runner._reconnect_failed_platform(platform, time.monotonic() + 1)
            drain = replacement._held_inbound_redispatch_task
            if drain is not None:
                await drain

        assert runner.adapters[platform] is replacement
        assert [c.args[0] for c in replacement.handle_message.await_args_list] == [held]
