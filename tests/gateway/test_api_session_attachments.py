"""AI Kosmo: authenticated API file parts become bounded, staged @file references."""
from pathlib import Path

import pytest

from gateway.platforms.api_server import _stage_session_file_parts
from hermes_constants import get_hermes_home


def test_session_file_parts_are_staged_on_the_persistent_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    content = _stage_session_file_parts(
        [
            {"type": "input_text", "text": "Read this"},
            {"type": "input_file", "filename": "../guest notes.txt", "file_data": "data:text/plain;base64,aGVsbG8="},
        ],
        "session/one",
    )

    assert content[0] == {"type": "input_text", "text": "Read this"}
    staged = Path(content[1]["text"].splitlines()[-1].removeprefix("@file:"))
    # Absolute, so a tool in another working directory reads it; under HERMES_HOME, so it survives
    # replacing the gateway's working directory.
    assert staged.is_absolute() and staged.parent == get_hermes_home() / "api-attachments" / "session-one"
    assert staged.read_text(encoding="utf-8") == "hello"
    assert staged.name.endswith("guest-notes.txt")
    assert not (tmp_path / ".hermes").exists()


def test_invalid_session_file_parts_leave_no_staged_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="valid base64"):
        _stage_session_file_parts(
            [{"type": "input_file", "filename": "notes.txt", "file_data": "data:text/plain;base64,not!base64"}],
            "session-one",
        )

    assert not (get_hermes_home() / "api-attachments" / "session-one").exists()
