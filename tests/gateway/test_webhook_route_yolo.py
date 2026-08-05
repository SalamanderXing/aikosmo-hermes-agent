"""Per-route ``yolo: true`` — unattended webhook lanes skip approvals.

A webhook route is an unattended lane: a pending approval has no human
behind it, times out, and auto-denies, killing the run. Routes that opt in
via ``yolo: true`` get session-scoped yolo for exactly the lifetime of the
delivery's agent run (hardline blocklist still applies inside yolo).
"""

from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.webhook import WebhookAdapter
from tools.approval import (
    disable_session_yolo,
    is_session_yolo_enabled,
)


def _make_adapter(routes):
    config = PlatformConfig(
        enabled=True,
        extra={"host": "127.0.0.1", "port": 0, "routes": routes},
    )
    return WebhookAdapter(config)


def _make_event(adapter, route_name, delivery_id):
    chat_id = f"webhook:{route_name}:{delivery_id}"
    source = adapter.build_source(
        chat_id=chat_id,
        chat_name=f"webhook/{route_name}",
        chat_type="webhook",
        user_id=f"webhook:{route_name}",
        user_name=route_name,
    )
    return MessageEvent(
        text="do the thing",
        message_type=MessageType.TEXT,
        source=source,
        raw_message={},
        message_id=delivery_id,
    )


def _attach_runner(adapter):
    runner = MagicMock()
    runner._session_key_for_source = (
        lambda source: f"agent:main:webhook:{source.chat_id}"
    )
    adapter.gateway_runner = runner
    return runner


@pytest.mark.asyncio
async def test_yolo_route_enables_and_disables_session_yolo():
    adapter = _make_adapter(
        {"auto-demo": {"secret": "s", "prompt": "p", "yolo": True}}
    )
    _attach_runner(adapter)
    event = _make_event(adapter, "auto-demo", "d-1")
    key = "agent:main:webhook:webhook:auto-demo:d-1"

    try:
        await adapter.on_processing_start(event)
        assert is_session_yolo_enabled(key)

        await adapter.on_processing_complete(event, outcome="success")
        assert not is_session_yolo_enabled(key)
    finally:
        disable_session_yolo(key)


@pytest.mark.asyncio
async def test_non_yolo_route_leaves_approvals_untouched():
    adapter = _make_adapter({"alerts": {"secret": "s", "prompt": "p"}})
    _attach_runner(adapter)
    event = _make_event(adapter, "alerts", "d-2")
    key = "agent:main:webhook:webhook:alerts:d-2"

    try:
        await adapter.on_processing_start(event)
        assert not is_session_yolo_enabled(key)
    finally:
        disable_session_yolo(key)


@pytest.mark.asyncio
async def test_yolo_route_without_runner_stays_safe():
    """No resolvable session key -> no yolo, no crash."""
    adapter = _make_adapter(
        {"auto-demo": {"secret": "s", "prompt": "p", "yolo": True}}
    )
    adapter.gateway_runner = None
    event = _make_event(adapter, "auto-demo", "d-3")

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, outcome="success")
