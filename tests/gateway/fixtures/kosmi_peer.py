"""AI Kosmo test peer: a loopback OpenAI-compatible model whose replies a test scripts per request.

``script(request)`` (a method: ``self.server`` is the test's handle) returns ``(message, finish_reason)`` for a chat request; ``BLOCK_FIFO`` in the newest
user message holds the request until ``server.release`` is set (``server.blocked`` says it arrived). Both
streamed and plain responses are written; every chat request is kept in ``server.requests``.
"""
import json
import re

from shared_authority_peer import ModelPeer


def newest_user_text(request) -> str:
    return json.dumps(next((m.get("content", "") for m in reversed(request.get("messages", []))
                            if m.get("role") == "user"), ""))


def tool_call(name, arguments, call_id="call_1"):
    return ({"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]},
        "tool_calls")


def text(content):
    return {"role": "assistant", "content": content}, "stop"


def tags_or_ack(request):
    """``ANSWER TAG-x`` for the newest user message's tag, ``NO_REPLY`` for SAY_NOTHING, else LOCAL_ACK."""
    user = newest_user_text(request)
    tag = re.findall(r"TAG-[a-z]+", user)
    return text("NO_REPLY" if "SAY_NOTHING" in user else f"ANSWER {tag[-1]}" if tag else "LOCAL_ACK")


class ScriptedPeer(ModelPeer):
    def script(self, request):
        return tags_or_ack(request)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "messages" not in request:
            return self._write(request, *text("ok"))
        if "BLOCK_FIFO" in newest_user_text(request) and request["messages"][-1].get("role") == "user":
            self.server.blocked.set()
            assert self.server.release.wait(60), "fixture model gate timed out"
        self.server.requests.append(request)
        self._write(request, *self.script(request))

    def _write(self, request, message, finish):
        base = {"id": "c", "model": "local-wire-stub", "created": 1}
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if request.get("stream") else "application/json")
        self.end_headers()
        if not request.get("stream"):
            self.wfile.write(json.dumps({**base, "object": "chat.completion", "choices": [
                {"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}).encode())
            return
        delta = {"role": "assistant"}
        if message.get("tool_calls"):
            delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]
        else:
            delta["content"] = message["content"]
        for chunk_delta, chunk_finish in ((delta, None), ({}, finish)):
            chunk = {**base, "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": chunk_delta, "finish_reason": chunk_finish}]}
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
