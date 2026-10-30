"""Customer Cloud tools: reads, this conversation's uploads, and kosmo-cli pinned to the customer's hotels. No shell, arbitrary URL, database or service credentials."""
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler

from gateway.session_context import (
    kosmo_attachment_root,
    kosmo_cli_access_token,
    kosmo_cli_restricted_hotel_slugs,
)
from tools.registry import registry


logger = logging.getLogger(__name__)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def cloud_read(args, **kwargs):
    token = kosmo_cli_access_token.get()
    if not token:
        return json.dumps({"error": "Please sign in to Cloud again."})
    resource = args.get("resource")
    slug = args.get("slug")
    if resource not in {"properties", "property", "knowledge"}:
        return json.dumps({"error": "Unknown read capability."})
    if resource != "properties" and (not isinstance(slug, str) or not slug):
        return json.dumps({"error": "Choose a property first."})
    query = {"resource": resource}
    if slug:
        query["slug"] = slug
    # Fixed trusted origin, never a model argument. Redirects cannot forward JWTs.
    origin = os.environ.get("KOSMO_CLI_INTERNAL_API_BASE_URL", "http://127.0.0.1:8080/internal").rstrip("/").removesuffix("/internal")
    request = Request(
        origin + "/staff/kosmi-assistant/read?" + urlencode(query),
        headers={"Authorization": "Bearer " + token, "Accept": "application/json"},
        method="GET",
    )
    try:
        with build_opener(NoRedirect()).open(request, timeout=20) as response:
            payload = response.read(100_001)
            if len(payload) > 100_000:
                logger.warning("Cloud read exceeded response limit: resource=%s", resource)
                return json.dumps({"error": "This result is too large. Ask about a smaller set of information."})
            return payload.decode("utf-8")
    except HTTPError as error:
        logger.warning("Cloud read failed: resource=%s status=%s", resource, error.code)
        if error.code in {401, 403, 404}:
            return json.dumps({"error": "This information is not available to your account. Sign in again if your session expired."})
        return json.dumps({"error": "Cloud information is temporarily unavailable. Please try again."})
    except (URLError, TimeoutError, UnicodeError) as error:
        logger.warning("Cloud read unavailable: resource=%s error_type=%s", resource, type(error).__name__)
        return json.dumps({"error": "Cloud information is temporarily unavailable. Please try again."})


registry.register(
    name="cloud_read",
    toolset="kosmo-cloud-customer",
    schema={
        "name": "cloud_read",
        "description": "Read your accessible AI Kosmo properties, a property's basic information, or the first page of its saved knowledge. Never changes data. Ask which property when unclear.",
        "parameters": {
            "type": "object",
            "properties": {
                "resource": {"type": "string", "enum": ["properties", "property", "knowledge"]},
                "slug": {"type": "string", "description": "Property slug from the current page or properties read; required for property and knowledge."},
            },
            "required": ["resource"],
            "additionalProperties": False,
        },
    },
    handler=cloud_read,
    description="Caller-bound customer Cloud reads",
)


def cloud_read_attachment(args, **kwargs):
    root = kosmo_attachment_root.get()
    if not root or not kosmo_cli_access_token.get():
        return json.dumps({"error": "Attachments are not available in this conversation."})
    path = args.get("path")
    offset = args.get("offset", 1)
    if not isinstance(path, str) or not path or type(offset) is not int or offset < 1:
        return json.dumps({"error": "Choose an attached file and a positive line offset."})
    try:
        candidate = Path(path)
        resolved = candidate.resolve()
        if candidate.is_symlink() or resolved.parent != Path(root).resolve() or not resolved.is_file():
            return json.dumps({"error": "This file is not an attachment in this conversation."})
        from tools.file_tools import read_file_tool
        result = json.loads(read_file_tool(str(resolved), offset=offset, task_id=kwargs.get("task_id", "default")))
        if "error" in result:
            logger.warning("Customer attachment extraction failed")
            return json.dumps({"error": "This attachment could not be read. Try a text version or a different file."})
        return json.dumps({key: result[key] for key in ("content", "total_lines", "truncated", "next_offset", "hint") if key in result})
    except (OSError, ValueError):
        logger.warning("Customer attachment read failed", exc_info=True)
        return json.dumps({"error": "This attachment could not be read. Try uploading it again."})


registry.register(
    name="cloud_read_attachment",
    toolset="kosmo-cloud-customer",
    schema={
        "name": "cloud_read_attachment",
        "description": "Read a document uploaded in this conversation. Use the path from its attached-file reference. Supports text, PDF and Office documents; images are visible directly. Only this conversation's uploads can be read. Never changes files. Follow pagination hints for large documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path from the uploaded document's @file reference."},
                "offset": {"type": "integer", "minimum": 1, "description": "First line to read, starting at 1."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    handler=cloud_read_attachment,
    description="Read documents uploaded in the current Cloud conversation",
)


# ---------------------------------------------------------------------------
# kosmo_cli: the staff CLI, pinned to the customer's own properties.
#
# No shell is involved: the command line is split into argv and `bun kosmo-cli`
# is spawned directly, in the customer profile (REST-only, caller-authenticated,
# hotel-guarded; see apps/backend/scripts/kosmo-cli/profile.ts). The process
# environment is rebuilt from an allowlist so the run's own credential and
# hotel scope are the only caller-specific values, and local paths are limited
# to this conversation's uploads.
# ---------------------------------------------------------------------------

_KOSMO_CLI_TIMEOUT_SECONDS = 600
_KOSMO_CLI_MAX_OUTPUT_CHARS = 60_000
# Non-secret host settings the CLI needs to boot and reach the API.
_KOSMO_CLI_PASSTHROUGH_ENV = (
    "PATH",
    "LANG",
    "LC_ALL",
    "TZ",
    "NODE_ENV",
    "BUN_INSTALL",
    "BUN_INSTALL_BIN",
    "BUN_RUNTIME_TRANSPILER_CACHE_PATH",
    # dotenvx needs the production key to decrypt the checked-in env file; the
    # customer profile never prints configuration and has no shell to read it.
    "DOTENV_PRIVATE_KEY_PRODUCTION",
    "KOSMO_CLI_INTERNAL_API_BASE_URL",
    "KOSMO_CLI_STAFF_API_BASE_URL",
)
_PATH_FLAGS = {"--content-file", "--value-file", "--answer-file", "--question-file", "--upload", "-o", "--output"}
_JWT_LIKE = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")


def _kosmo_cli_repo() -> str:
    return os.environ.get("HERMES_CWD") or os.environ.get("KOSMO_CLI_REPO", "/data/repo")


def _kosmo_cli_reject_paths(argv: list[str], attachment_root: str | None) -> str | None:
    """Only this conversation's uploads may be referenced as local files."""
    allowed_root = Path(attachment_root).resolve() if attachment_root else None
    for index, token in enumerate(argv):
        previous = argv[index - 1] if index > 0 else ""
        looks_like_path = token.startswith(("/", "./", "../", "~")) or previous in _PATH_FLAGS
        if not looks_like_path:
            continue
        if previous in {"-o", "--output"}:
            return "Saving to files is not available in this conversation."
        candidate = Path(token).expanduser()
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            return "That file is not an attachment in this conversation."
        if (
            allowed_root is None
            or candidate.is_symlink()
            or resolved.parent != allowed_root
            or not resolved.is_file()
        ):
            return "That file is not an attachment in this conversation. Upload it here first."
    return None


def _kosmo_cli_env(token: str, slugs: tuple[str, ...], home: str) -> dict[str, str]:
    env = {key: os.environ[key] for key in _KOSMO_CLI_PASSTHROUGH_ENV if os.environ.get(key)}
    env.update(
        {
            "HOME": home,
            "NO_COLOR": "1",
            "FORCE_COLOR": "0",
            "I18NEXT_NO_SUPPORT_NOTICE": "1",
            "KOSMO_CLI_PROFILE": "customer",
            "KOSMO_CLI_AUTH_ACCESS_TOKEN": token,
            "KOSMO_CLI_RESTRICTED_HOTEL_SLUGS": ",".join(slugs),
        }
    )
    return env


def kosmo_cli(args, **kwargs):
    token = kosmo_cli_access_token.get()
    slugs = kosmo_cli_restricted_hotel_slugs.get()
    if not token or slugs is None:
        return json.dumps({"error": "Please sign in to Cloud again."})
    if not slugs:
        return json.dumps({"error": "No property is linked to this account yet. Contact AI Kosmo support."})
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return json.dumps({"error": "Give the kosmo-cli command to run, e.g. `hotel <slug> knowledge`."})
    try:
        argv = shlex.split(command)
    except ValueError:
        return json.dumps({"error": "That command line could not be parsed. Check the quotes."})
    # Accept the documented spelling as well as bare arguments.
    if argv[:2] == ["bun", "kosmo-cli"]:
        argv = argv[2:]
    elif argv[:1] == ["kosmo-cli"]:
        argv = argv[1:]
    if not argv:
        return json.dumps({"error": "Give the kosmo-cli command to run, e.g. `hotel <slug> knowledge`."})
    if any("\x00" in token_ for token_ in argv):
        return json.dumps({"error": "Invalid command."})
    rejection = _kosmo_cli_reject_paths(argv, kosmo_attachment_root.get())
    if rejection:
        return json.dumps({"error": rejection})
    repo = _kosmo_cli_repo()
    with tempfile.TemporaryDirectory(prefix="kosmi-customer-cli-") as home:
        try:
            completed = subprocess.run(
                ["bun", "kosmo-cli", *argv],
                cwd=repo,
                env=_kosmo_cli_env(token, slugs, home),
                capture_output=True,
                text=True,
                timeout=_KOSMO_CLI_TIMEOUT_SECONDS,
                stdin=subprocess.DEVNULL,
                check=False,  # the exit code is part of the result the model reads
            )
        except subprocess.TimeoutExpired:
            logger.warning("Customer kosmo-cli timed out: %s", argv[:2])
            return json.dumps({"error": "This request took too long. Try a narrower question."})
        except OSError:
            logger.warning("Customer kosmo-cli could not start", exc_info=True)
            return json.dumps({"error": "The property tools are temporarily unavailable. Please try again."})
    output = _scrub_kosmo_cli_output((completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else ""))
    truncated = len(output) > _KOSMO_CLI_MAX_OUTPUT_CHARS
    if truncated:
        output = output[:_KOSMO_CLI_MAX_OUTPUT_CHARS]
    return json.dumps(
        {
            "exit_code": completed.returncode,
            "output": output.strip(),
            **({"truncated": True, "hint": "Narrow the command (add --limit, a query, or --json) to see the rest."} if truncated else {}),
        }
    )


_NOISE_LINES = (
    "[MISSING_ENV_FILE]",
    "[dotenvx@",
    "i18next is made possible",
    "[VoiceConfig]",
    "Command failed with exit code",
    'error: script "kosmo-cli" exited',
)


def _scrub_kosmo_cli_output(text: str) -> str:
    lines = [line for line in text.splitlines() if not any(marker in line for marker in _NOISE_LINES)]
    return _JWT_LIKE.sub("[redacted]", "\n".join(lines))


registry.register(
    name="kosmo_cli",
    toolset="kosmo-cloud-customer",
    schema={
        "name": "kosmo_cli",
        "description": (
            "Run one AI Kosmo property command for this customer (their properties only). "
            "Start with `help` to see the available commands, then read (`hotel <slug> knowledge`, `hotel <slug> chats`, "
            "`skills show <slug> <name>` ...) before changing anything (`hotel <slug> kosmo-web --set ...`, `skills patch ...`, "
            "`qa-documents add ...`, `hotel <slug> knowledge source upload <attachment path>`). Add --json for machine-readable output. "
            "Never describe the command itself to the customer; explain the outcome in their terms."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Arguments after `bun kosmo-cli`, e.g. `hotel my-hotel knowledge` or `skills patch my-hotel my-hotel/pets --old \"...\" --new \"...\"`.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    handler=kosmo_cli,
    description="Run kosmo-cli for the customer's own properties",
)
