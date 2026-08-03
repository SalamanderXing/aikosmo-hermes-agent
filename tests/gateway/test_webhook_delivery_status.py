from pathlib import Path

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
