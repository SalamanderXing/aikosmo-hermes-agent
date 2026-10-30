"""Exercise upload staging and the real agent-to-database persistence boundary."""
import base64
import json
import shutil

import pytest

from gateway.platforms.api_server import APIServerAdapter, _stage_session_file_parts
from hermes_state import SessionDB
from tests.agent.test_run_agent import agent
from tools.file_tools import read_file_tool


@pytest.mark.parametrize("platform", ["api_server", "discord"])
def test_image_flush_reload_and_client_projection(agent, tmp_path, platform):
    db = SessionDB(db_path=tmp_path / "uploads.db")
    db.create_session("upload-history", source=platform)
    agent.platform = platform
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = "upload-history"
    content = [
        {"type": "text", "text": "Read this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}},
    ]
    try:
        agent._flush_messages_to_session_db([{"role": "user", "content": content}], [])
        rows = db.get_messages("upload-history")
        assert len(rows) == 1
        expected = content if platform == "api_server" else "Read this image\n[screenshot]"
        assert APIServerAdapter._message_response(rows[0])["content"] == expected
        assert db.get_messages_as_conversation("upload-history")[0]["content"] == expected
    finally:
        agent._session_db = None
        db.close()


def test_staged_document_survives_gateway_replacement_and_a_different_tool_directory(tmp_path, monkeypatch):
    gateway_cwd = tmp_path / "gateway"
    tool_cwd = tmp_path / "worktree"
    gateway_cwd.mkdir()
    tool_cwd.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "persistent-home"))
    monkeypatch.chdir(gateway_cwd)
    monkeypatch.setenv("TERMINAL_CWD", str(tool_cwd))
    [part] = _stage_session_file_parts([{
        "type": "input_file", "filename": "note.txt",
        "file_data": "data:text/plain;base64," + base64.b64encode(b"UPLOAD_SENTINEL").decode(),
    }], "upload-doc")
    path = part["text"].split("\n@file:", 1)[1]
    monkeypatch.chdir(tool_cwd)
    shutil.rmtree(gateway_cwd)
    result = json.loads(read_file_tool(path))
    assert not result.get("error")
    assert "UPLOAD_SENTINEL" in result["content"]
