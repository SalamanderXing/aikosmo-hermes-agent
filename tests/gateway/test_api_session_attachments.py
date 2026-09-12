"""AI Kosmo: authenticated API file parts become bounded, staged @file references."""
from pathlib import Path

import pytest

from gateway.platforms.api_server import _stage_session_file_parts


def test_session_file_parts_are_staged_as_workspace_references(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    content = _stage_session_file_parts(
        [
            {"type": "input_text", "text": "Read this"},
            {"type": "input_file", "filename": "../guest notes.txt", "file_data": "data:text/plain;base64,aGVsbG8="},
        ],
        "session/one",
    )

    assert content[0] == {"type": "input_text", "text": "Read this"}
    reference = content[1]["text"].splitlines()[-1].removeprefix("@file:")
    staged = tmp_path / reference
    assert staged.read_text(encoding="utf-8") == "hello"
    assert staged.parent.name == "session-one"
    assert staged.name.endswith("guest-notes.txt")


def test_invalid_session_file_parts_leave_no_staged_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="valid base64"):
        _stage_session_file_parts(
            [{"type": "input_file", "filename": "notes.txt", "file_data": "data:text/plain;base64,not!base64"}],
            "session-one",
        )

    assert not (tmp_path / ".hermes").exists()
