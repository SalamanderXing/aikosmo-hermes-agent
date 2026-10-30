"""AI Kosmo customer tools: caller-bound Cloud reads, the upload reader and the scoped kosmo-cli.

The runtime that selects them (effective tools, prompt, memory, credentials per run, restart) is
covered end to end by ``test_kosmi_cloud_runs.py``.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch


from gateway.session_context import kosmo_attachment_root, kosmo_cli_access_token


def test_read_tool_keeps_caller_tokens_separate_and_never_follows_redirects():
    from tools.kosmo_cloud_tool import cloud_read, NoRedirect
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, limit): return b'{"properties": []}'
    def open_request(request, timeout):
        calls.append((request.full_url, request.get_header("Authorization"), request.method))
        return Response()
    def call(token):
        reset = kosmo_cli_access_token.set(token)
        try:
            return cloud_read({"resource": "properties", "url": "https://attacker.invalid"})
        finally:
            kosmo_cli_access_token.reset(reset)
    with patch("tools.kosmo_cloud_tool.build_opener") as opener:
        opener.return_value.open.side_effect = open_request
        assert "sign in" in cloud_read({"resource": "properties"}).lower()
        opener.assert_not_called()
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert len(list(pool.map(call, ["alice", "bob"]))) == 2
    assert {auth for _, auth, _ in calls} == {"Bearer alice", "Bearer bob"}
    assert all("attacker" not in url and method == "GET" for url, _, method in calls)
    assert NoRedirect().redirect_request(None, None, 302, "", {}, "https://attacker.invalid") is None


def test_customer_lifecycle_cannot_inject_staff_plugin_context():
    from hermes_cli.lifecycle import invoke_hook
    with patch("hermes_cli.plugins.invoke_hook", return_value=[{"context": "STAFF_PLUGIN_SENTINEL"}]) as plugin:
        assert invoke_hook("pre_llm_call", platform="cloud_customer") == []
        plugin.assert_not_called()
        assert invoke_hook("pre_llm_call", platform="api_server") == [{"context": "STAFF_PLUGIN_SENTINEL"}]


def test_attachment_reader_scope_and_document_extraction(tmp_path, monkeypatch):
    from gateway.platforms.api_server import session_attachment_directory
    from gateway.session_context import kosmo_attachment_root
    from tools.kosmo_cloud_tool import cloud_read_attachment
    from tests.tools.test_read_extract import _write_docx
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    root = session_attachment_directory("alice")
    root.mkdir(parents=True)
    document = root / "0123456789-notes.docx"
    _write_docx(document, '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Document sentinel</w:t></w:r></w:p></w:body></w:document>')
    text = root / "1234567890-notes.txt"
    text.write_text("first\nsecond\nthird")
    outside = tmp_path / "staff.txt"
    outside.write_text("STAFF_SECRET")
    other = session_attachment_directory("bob")
    other.mkdir()
    (other / text.name).write_text("OTHER_CUSTOMER_SECRET")
    link = root / "0123456789-link.txt"
    link.symlink_to(outside)
    assert "error" in json.loads(cloud_read_attachment({"path": str(text)}))
    token = kosmo_cli_access_token.set("alice-token")
    scope = kosmo_attachment_root.set(str(root))
    try:
        result = json.loads(cloud_read_attachment({"path": str(document)}))
        assert "Document sentinel" in result["content"], result
        assert str(tmp_path) not in json.dumps(result)
        assert "first" not in json.loads(cloud_read_attachment({"path": str(text), "offset": 2}))["content"]
        for path in (outside, other / text.name, link, root / "missing.txt", root):
            result = cloud_read_attachment({"path": str(path)})
            assert "error" in json.loads(result)
            assert "SECRET" not in result
            assert str(tmp_path) not in result
        for offset in (0, -1, "1", True):
            assert "error" in json.loads(cloud_read_attachment({"path": str(text), "offset": offset}))
        malformed = root / "0123456789-broken.docx"
        malformed.write_text("not a document")
        assert "error" in json.loads(cloud_read_attachment({"path": str(malformed)}))
    finally:
        kosmo_attachment_root.reset(scope)
        kosmo_cli_access_token.reset(token)
    assert "error" in json.loads(cloud_read_attachment({"path": str(text)}))


def test_concurrent_attachment_readers_cannot_cross_sessions(tmp_path, monkeypatch):
    import threading
    from gateway.platforms.api_server import session_attachment_directory
    from gateway.session_context import kosmo_attachment_root
    from tools.kosmo_cloud_tool import cloud_read_attachment
    monkeypatch.chdir(tmp_path)
    roots = [session_attachment_directory(name) for name in ("alice", "bob")]
    for index, root in enumerate(roots):
        root.mkdir(parents=True)
        (root / "0123456789-notes.txt").write_text(f"CUSTOMER_{index}")
    barrier = threading.Barrier(2)
    def read(index):
        scope = kosmo_attachment_root.set(str(roots[index]))
        token = kosmo_cli_access_token.set(f"token-{index}")
        try:
            barrier.wait(timeout=5)
            own = json.loads(cloud_read_attachment({"path": str(roots[index] / "0123456789-notes.txt")}))
            other = json.loads(cloud_read_attachment({"path": str(roots[1-index] / "0123456789-notes.txt")}))
            assert f"CUSTOMER_{index}" in own["content"]
            assert "error" in other
        finally:
            kosmo_attachment_root.reset(scope)
            kosmo_cli_access_token.reset(token)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(read, (0, 1)))
    assert kosmo_attachment_root.get() is None


def test_kosmo_cli_tool_runs_customer_profile_with_scoped_env(tmp_path, monkeypatch):
    from gateway.session_context import kosmo_cli_restricted_hotel_slugs
    from tools import kosmo_cloud_tool

    monkeypatch.setenv("HERMES_CWD", str(tmp_path))
    monkeypatch.setenv("DOTENV_PRIVATE_KEY_PRODUCTION", "dotenv-key")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "must-not-leak")
    monkeypatch.setenv("MONITOR_API_KEY", "must-not-leak")
    monkeypatch.setenv("KOSMO_CLI_AUTH_EMAIL", "operator@example.com")
    monkeypatch.setenv("KOSMO_CLI_AUTH_PASSWORD", "must-not-leak")
    monkeypatch.setenv("HOME", "/root")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess_result(0, "Kosmo Web: my-hotel\nintro_message: hi eyJabcdefghijklmnop.qrstuvwxyz0123456789.ABCDEFGHIJKLMNOPQR\n", "🌐 i18next is made possible by our own product\n")

    with patch("tools.kosmo_cloud_tool.subprocess.run", side_effect=fake_run):
        assert json.loads(kosmo_cloud_tool.kosmo_cli({"command": "hotel my-hotel kosmo-web"}))["error"]
        token = kosmo_cli_access_token.set("caller-jwt")
        scope = kosmo_cli_restricted_hotel_slugs.set(("my-hotel", "second"))
        try:
            result = json.loads(kosmo_cloud_tool.kosmo_cli({"command": "bun kosmo-cli hotel my-hotel kosmo-web"}))
        finally:
            kosmo_cli_access_token.reset(token)
            kosmo_cli_restricted_hotel_slugs.reset(scope)
    assert result["exit_code"] == 0
    assert "Kosmo Web: my-hotel" in result["output"]
    assert "[redacted]" in result["output"] and "eyJabcdefghijklmnop" not in result["output"]
    assert "i18next" not in result["output"]
    argv, kwargs = calls[0]
    assert argv == ["bun", "kosmo-cli", "hotel", "my-hotel", "kosmo-web"]
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["stdin"] is not None
    env = kwargs["env"]
    assert env["KOSMO_CLI_PROFILE"] == "customer"
    assert env["KOSMO_CLI_AUTH_ACCESS_TOKEN"] == "caller-jwt"
    assert env["KOSMO_CLI_RESTRICTED_HOTEL_SLUGS"] == "my-hotel,second"
    assert env["DOTENV_PRIVATE_KEY_PRODUCTION"] == "dotenv-key"
    assert env["HOME"] != "/root"
    for secret in ("SUPABASE_JWT_SECRET", "MONITOR_API_KEY", "KOSMO_CLI_AUTH_EMAIL", "KOSMO_CLI_AUTH_PASSWORD"):
        assert secret not in env


def subprocess_result(code, out, err):
    import subprocess

    return subprocess.CompletedProcess(args=[], returncode=code, stdout=out, stderr=err)


def test_kosmo_cli_tool_only_accepts_this_conversations_uploads(tmp_path, monkeypatch):
    from gateway.session_context import kosmo_cli_restricted_hotel_slugs
    from tools import kosmo_cloud_tool

    uploads = tmp_path / "api-attachments" / "kosmi_one"
    uploads.mkdir(parents=True)
    own = uploads / "rules.md"
    own.write_text("pool rules")
    other = tmp_path / "api-attachments" / "kosmi_two" / "secret.md"
    other.parent.mkdir(parents=True)
    other.write_text("not yours")
    link = uploads / "link.md"
    link.symlink_to(other)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess_result(0, "ok", "")

    token = kosmo_cli_access_token.set("caller-jwt")
    scope = kosmo_cli_restricted_hotel_slugs.set(("my-hotel",))
    root = kosmo_attachment_root.set(str(uploads))
    try:
        with patch("tools.kosmo_cloud_tool.subprocess.run", side_effect=fake_run):
            ok = json.loads(kosmo_cloud_tool.kosmo_cli({"command": f"hotel my-hotel knowledge source upload {own}"}))
            assert ok["exit_code"] == 0
            for command in (
                f"hotel my-hotel knowledge source upload {other}",
                f"hotel my-hotel knowledge source upload {link}",
                "skills edit my-hotel my-hotel/pets --content-file /data/repo/.env.production",
                "skills edit my-hotel my-hotel/pets --content-file ~/.kosmo-cli/auth-sessions.json",
                "hotel my-hotel kosmo-web --upload logo_url=/etc/passwd",
                "system-prompt my-hotel -o /tmp/out.md",
                "hotel my-hotel knowledge source upload ../kosmi_two/secret.md",
            ):
                assert "error" in json.loads(kosmo_cloud_tool.kosmo_cli({"command": command})), command
        assert len(calls) == 1
        assert json.loads(kosmo_cloud_tool.kosmo_cli({"command": "hotel 'unterminated"}))["error"]
        assert json.loads(kosmo_cloud_tool.kosmo_cli({"command": "   "}))["error"]
    finally:
        kosmo_attachment_root.reset(root)
        kosmo_cli_restricted_hotel_slugs.reset(scope)
        kosmo_cli_access_token.reset(token)
    assert kosmo_cli_restricted_hotel_slugs.get() is None


def test_kosmo_cli_tool_refuses_without_hotel_scope():
    from gateway.session_context import kosmo_cli_restricted_hotel_slugs
    from tools import kosmo_cloud_tool

    token = kosmo_cli_access_token.set("caller-jwt")
    try:
        assert "sign in" in json.loads(kosmo_cloud_tool.kosmo_cli({"command": "help"}))["error"].lower()
        scope = kosmo_cli_restricted_hotel_slugs.set(())
        try:
            assert "No property" in json.loads(kosmo_cloud_tool.kosmo_cli({"command": "help"}))["error"]
        finally:
            kosmo_cli_restricted_hotel_slugs.reset(scope)
    finally:
        kosmo_cli_access_token.reset(token)


def test_shell_snapshots_never_capture_the_caller_token(tmp_path):
    import subprocess
    from tools.environments.base_session_env import _export_dump_excluding_session_vars
    from tools.environments.local import _make_run_env
    token = kosmo_cli_access_token.set("snapshot-caller-token")
    try:
        env = _make_run_env({})
        assert env["KOSMO_CLI_AUTH_ACCESS_TOKEN"] == "snapshot-caller-token"
        snapshot = tmp_path / "snapshot.env"
        subprocess.run(["bash", "-c", _export_dump_excluding_session_vars(str(snapshot))], env=env, check=True)
    finally:
        kosmo_cli_access_token.reset(token)
    assert "snapshot-caller-token" not in snapshot.read_text()
    assert "KOSMO_CLI_AUTH_ACCESS_TOKEN" not in _make_run_env({"KOSMO_CLI_AUTH_ACCESS_TOKEN": "stale"})
