import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import make_mocked_request

import gateway.platforms.api_server as api_server_module
from gateway.platforms.api_server import APIServerAdapter
from gateway.config import PlatformConfig
from gateway.webhook_deliveries import WebhookDeliveryStore


def test_delivery_is_idempotent_across_store_restart(tmp_path: Path) -> None:
    path = tmp_path / "deliveries.db"
    first = WebhookDeliveryStore(path)
    delivery, created = first.accept(
        route="auto-demo", delivery_id="delivery-1", request_id="request-1"
    )
    assert created is True
    assert delivery["status"] == "accepted"

    restarted = WebhookDeliveryStore(path)
    duplicate, created = restarted.accept(
        route="auto-demo", delivery_id="delivery-2", request_id="request-1"
    )
    assert created is False
    assert duplicate["delivery_id"] == "delivery-1"


def test_delivery_status_has_no_prompt_or_message_content(tmp_path: Path) -> None:
    store = WebhookDeliveryStore(tmp_path / "deliveries.db")
    store.accept(
        route="auto-demo", delivery_id="delivery-1", request_id="request-1"
    )
    store.mark_running("delivery-1", "session-1")
    store.mark_ended(
        "delivery-1", session_id="session-1", end_reason="success"
    )

    delivery = store.get("delivery-1")
    assert delivery is not None
    assert delivery["status"] == "ended"
    assert delivery["session_id"] == "session-1"
    assert delivery["end_reason"] == "success"
    assert "prompt" not in delivery
    assert "content" not in delivery


def test_api_server_registers_authenticated_delivery_status_route() -> None:
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "a" * 32})
    )
    routes = {(method, path) for method, path, _ in adapter._http_route_table()}
    assert ("GET", "/api/webhook-deliveries/{delivery_id}") in routes


def test_delivery_status_only_excludes_agent_control_routes() -> None:
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"key": "a" * 32, "delivery_status_only": True},
        )
    )
    routes = {(method, path) for method, path, _ in adapter._http_route_table()}

    assert ("GET", "/api/webhook-deliveries/{delivery_id}") in routes
    assert ("GET", "/api/webhook-deliveries/{delivery_id}/session") in routes
    assert ("POST", "/api/webhook-deliveries/{delivery_id}/feedback") in routes
    assert ("POST", "/api/sessions") not in routes
    assert ("POST", "/v1/runs") not in routes
    assert ("POST", "/api/jobs") not in routes


@pytest.mark.asyncio
async def test_delivery_session_returns_only_the_bound_auto_demo_transcript(monkeypatch) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    delivery_store = SimpleNamespace(
        get=lambda delivery_id: {
            "delivery_id": delivery_id,
            "route": "auto-demo",
            "session_id": "session-1",
        }
    )
    monkeypatch.setattr(
        api_server_module, "get_webhook_delivery_store", lambda: delivery_store
    )

    class SessionDB:
        def get_session(self, session_id):
            return {"id": session_id, "title": "Demo run", "started_at": 1.0}

        def get_messages(self, session_id):
            return [{"id": 1, "session_id": session_id, "role": "assistant", "content": "Ready"}]

    async def session_db():
        return SessionDB()

    adapter._ensure_session_db_async = session_db
    response = await adapter._handle_webhook_delivery_session(
        make_mocked_request("GET", "/", match_info={"delivery_id": "delivery-1"})
    )

    payload = json.loads(response.text)
    assert response.status == 200
    assert payload["session"]["id"] == "session-1"
    assert payload["messages"][0]["content"] == "Ready"


@pytest.mark.asyncio
async def test_feedback_reopens_and_reends_the_delivery_bound_session(monkeypatch) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    delivery_store = SimpleNamespace(
        get=lambda _delivery_id: {
            "delivery_id": "delivery-1",
            "route": "auto-demo",
            "session_id": "session-1",
        }
    )
    monkeypatch.setattr(
        api_server_module, "get_webhook_delivery_store", lambda: delivery_store
    )
    lifecycle = []

    class SessionDB:
        def get_session(self, session_id):
            return {"id": session_id}

        def reopen_session(self, session_id):
            lifecycle.append(("reopen", session_id))

        def end_session(self, session_id, reason):
            lifecycle.append(("end", session_id, reason))

    async def session_db():
        return SessionDB()

    async def chat_turn(self, request):
        assert request.match_info["session_id"] == "session-1"
        return api_server_module.web.json_response({"message": {"content": "Changed"}})

    adapter._ensure_session_db_async = session_db
    monkeypatch.setattr(
        APIServerAdapter._handle_session_chat, "__wrapped__", chat_turn
    )
    response = await adapter._handle_webhook_delivery_feedback(
        make_mocked_request("POST", "/", match_info={"delivery_id": "delivery-1"})
    )

    assert response.status == 200
    assert lifecycle == [
        ("reopen", "session-1"),
        ("end", "session-1", "webhook_feedback_complete"),
    ]
