"""Durable, content-free status records for webhook agent deliveries."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home


class WebhookDeliveryStore:
    """Small SQLite store shared by the webhook and API-server adapters."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (get_hermes_home() / "webhook-deliveries.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS webhook_deliveries (
                    delivery_id TEXT PRIMARY KEY,
                    route TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('accepted', 'running', 'ended')),
                    session_id TEXT,
                    accepted_at REAL NOT NULL,
                    started_at REAL,
                    ended_at REAL,
                    end_reason TEXT,
                    UNIQUE (route, request_id)
                )
                """
            )

    @staticmethod
    def _as_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        return dict(row) if row is not None else None

    def accept(
        self, *, route: str, delivery_id: str, request_id: str
    ) -> tuple[dict[str, Any], bool]:
        """Persist acceptance, returning the existing row for a duplicate."""
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO webhook_deliveries (
                    delivery_id, route, request_id, status, accepted_at
                ) VALUES (?, ?, ?, 'accepted', ?)
                """,
                (delivery_id, route, request_id, time.time()),
            )
            created = self._conn.execute("SELECT changes()").fetchone()[0] == 1
            row = self._conn.execute(
                "SELECT * FROM webhook_deliveries WHERE route = ? AND request_id = ?",
                (route, request_id),
            ).fetchone()
        assert row is not None
        return dict(row), created

    def mark_running(self, delivery_id: str, session_id: Optional[str] = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE webhook_deliveries
                SET status = CASE WHEN status = 'ended' THEN status ELSE 'running' END,
                    started_at = COALESCE(started_at, ?),
                    session_id = COALESCE(?, session_id)
                WHERE delivery_id = ?
                """,
                (time.time(), session_id, delivery_id),
            )

    def mark_ended(
        self, delivery_id: str, *, session_id: Optional[str], end_reason: str
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE webhook_deliveries
                SET status = 'ended', session_id = COALESCE(?, session_id),
                    started_at = COALESCE(started_at, ?), ended_at = ?, end_reason = ?
                WHERE delivery_id = ?
                """,
                (session_id, time.time(), time.time(), end_reason, delivery_id),
            )

    def get(self, delivery_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM webhook_deliveries WHERE delivery_id = ?",
                (delivery_id,),
            ).fetchone()
        return self._as_dict(row)


_stores: dict[str, WebhookDeliveryStore] = {}
_stores_lock = threading.Lock()


def get_webhook_delivery_store(path: Optional[Path] = None) -> WebhookDeliveryStore:
    key = str(path or (get_hermes_home() / "webhook-deliveries.db"))
    with _stores_lock:
        store = _stores.get(key)
        if store is None:
            store = WebhookDeliveryStore(path)
            _stores[key] = store
        return store
