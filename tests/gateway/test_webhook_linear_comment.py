"""AI Kosmo: posting a Linear comment exactly once (gateway/platforms/webhook_linear.py).

The lane end to end (admission, thread ordering, restart) is ``test_webhook_linear_lane.py``; these pin
the sender: one create per final answer keyed on the delivery id, read-back convergence after an
ambiguous failure, and every malformed input failing closed before Linear is called.
"""
import asyncio

import pytest

from gateway.config import PlatformConfig
from gateway.platforms import webhook_linear
from gateway.platforms.webhook import WebhookAdapter
from gateway.platforms.webhook_delivery import validate_destination
from hermes_state_runtime import RuntimeStoreError

DELIVERY = "7c3f1a6e-5b2d-4f8a-9c41-2e6b8d0a1f34"
ISSUE = "1eba60c7-9663-427c-b313-d588f33c77a8"
PARENT = "4a43b9b7-0aa8-46a3-9445-37eb40f8d98c"


def destination(**extra):
    return {"deliver": "linear_comment", "reply_id": extra.pop("reply_id", DELIVERY),
            "deliver_extra": {"issue_id": ISSUE, "parent_id": PARENT, **extra}}


def existing(**over):
    return {"id": DELIVERY, "body": "The answer.", "parentId": PARENT, "issue": {"id": ISSUE}, **over}


class Linear:
    """A small fake Linear: read-backs answer ``existing``, creates are recorded."""

    def __init__(self, monkeypatch, *, response=None, raises=None, existing=None, readback_raises=None,
                 readback_response=None):
        self.creates, self.readbacks = [], []
        self.response, self.raises, self.existing = response, raises, existing
        self.readback_raises, self.readback_response = readback_raises, readback_response
        monkeypatch.setattr(webhook_linear, "_linear_graphql", self)
        monkeypatch.setenv("LINEAR_API_KEY", "lin_api_test")

    def __call__(self, query, variables, api_key):
        assert api_key == "lin_api_test"
        if "KosmiCommentReadback" in query:
            self.readbacks.append(variables)
            if self.readback_raises is not None:
                raise self.readback_raises
            return self.readback_response or {"data": {"comment": self.existing}}
        self.creates.append(variables["input"])
        assert "DROP" not in query, "the body travels as a variable, never as query text"
        if self.raises is not None:
            raise self.raises
        return self.response or {"data": {"commentCreate": {"success": True, "comment": {"id": "new"}}}}


def deliver(content="The answer.", **extra):
    return asyncio.run(webhook_linear.deliver_linear_comment(content, destination(**extra)))


def test_the_final_answer_is_one_threaded_comment_keyed_on_the_delivery(monkeypatch):
    linear = Linear(monkeypatch)
    result = deliver('"); DROP { } $var \n end')
    assert result.success and result.message_id is None  # no message id: the stream consumer never edits
    assert linear.creates == [{"id": DELIVERY, "issueId": ISSUE, "body": '"); DROP { } $var \n end', "parentId": PARENT}]


def test_a_top_level_comment_omits_the_parent(monkeypatch):
    linear = Linear(monkeypatch)
    assert deliver(parent_id="").success
    assert "parentId" not in linear.creates[0]


@pytest.mark.parametrize("kwargs,error", [
    ({"response": {"errors": [{"message": "Entity not found"}]}}, "Entity not found"),
    ({"response": {"data": {"commentCreate": {"success": False}}}}, ""),
    ({"raises": OSError("connection reset")}, "connection reset"),
])
def test_failures_are_reported_as_failed_sends(monkeypatch, kwargs, error):
    Linear(monkeypatch, **kwargs)
    result = deliver()
    assert not result.success and error in (result.error or "")


@pytest.mark.parametrize("extra", [
    {"issue_id": ""}, {"issue_id": "not-a-uuid"}, {"issue_id": "{issue_id}"}, {"parent_id": "../../etc/passwd"},
    {"reply_id": ""}, {"reply_id": "1757764800000"},
    {"reply_id": "7c3f1a6e-5b2d-1f8a-9c41-2e6b8d0a1f34"},  # v1, not v4
    {"reply_id": "7c3f1a6e-5b2d-4f8a-0c41-2e6b8d0a1f34"},  # invalid variant
])
def test_malformed_ids_never_reach_linear(monkeypatch, extra):
    linear = Linear(monkeypatch)
    assert not deliver(**extra).success
    assert linear.creates == [] and linear.readbacks == []


def test_a_missing_api_key_never_reaches_linear(monkeypatch):
    linear = Linear(monkeypatch)
    monkeypatch.delenv("LINEAR_API_KEY")
    assert not deliver().success and linear.creates == []


def test_a_retry_of_a_landed_delivery_posts_nothing(monkeypatch):
    linear = Linear(monkeypatch, existing=existing())
    assert deliver().success and linear.creates == []


def test_a_post_commit_timeout_converges_on_the_same_comment(monkeypatch):
    linear = Linear(monkeypatch, raises=TimeoutError("read timed out"))
    linear.existing = None

    def commit_then_fail(query, variables, api_key):
        if "commentCreate" in query:
            linear.existing = existing()
        return Linear.__call__(linear, query, variables, api_key)
    monkeypatch.setattr(webhook_linear, "_linear_graphql", commit_then_fail)
    assert deliver().success, "a committed comment must not read as failure"
    assert deliver().success and len(linear.creates) == 1, "the retry must not post again"


@pytest.mark.parametrize("mismatch", [
    {"body": "somebody else's text"}, {"issue": {"id": "99999999-9999-4999-8999-999999999999"}},
    {"parentId": "11111111-1111-4111-8111-111111111111"}, {"id": "22222222-2222-4222-8222-222222222222"},
])
def test_a_different_comment_under_our_id_fails_closed(monkeypatch, mismatch):
    Linear(monkeypatch, raises=TimeoutError("ambiguous"), existing=existing(**mismatch))
    assert not deliver().success


def test_an_unreadable_or_partial_read_back_never_proves_success(monkeypatch):
    Linear(monkeypatch, raises=TimeoutError("ambiguous"), readback_raises=OSError("still unreachable"))
    assert not deliver().success
    linear = Linear(monkeypatch, raises=TimeoutError("ambiguous"), readback_response={
        "data": {"comment": existing()}, "errors": [{"message": "Failed to resolve parentId"}]})
    assert not deliver().success
    assert len(linear.creates) == 1 and len(linear.readbacks) == 2


def test_the_destination_persists_only_the_ids_the_sender_reads():
    stored = validate_destination({"deliver": "linear_comment", "toolset_tier": "staff", "deliver_extra": {
        "issue_id": ISSUE, "parent_id": PARENT, "trigger": "mention", "comment_body": "secret customer text"}})
    assert stored == {"deliver": "linear_comment", "toolset_tier": "staff",
                      "deliver_extra": {"issue_id": ISSUE, "parent_id": PARENT, "trigger": "mention"}}
    # Other destinations are byte-identical to upstream's (retry fingerprints): no tier key appears.
    assert validate_destination({"deliver": "log", "deliver_extra": {}}) == {"deliver": "log", "deliver_extra": {}}
    for bad in ({"issue_id": 7}, {"trigger": ["x"]}):
        with pytest.raises(RuntimeStoreError):
            validate_destination({"deliver": "linear_comment", "deliver_extra": bad})


@pytest.fixture
def adapter(monkeypatch):
    linear = Linear(monkeypatch)
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"routes": {}}))
    targets = {}
    monkeypatch.setattr(webhook_linear, "turn_destination", lambda _a, chat_id, turn=None: targets[chat_id])
    import gateway.platforms.webhook as webhook_module
    monkeypatch.setattr(webhook_module, "turn_destination", lambda _a, chat_id, turn=None: targets[chat_id])
    adapter.targets, adapter.linear = targets, linear
    return adapter


@pytest.mark.parametrize("metadata", [None, {}, {"notify": False}, {"notify": None}, {"expect_edits": True},
                                      {"notify": "yes"}, {"notify": 1}])
def test_only_a_notify_final_posts(adapter, metadata):
    adapter.targets["chat"] = destination()
    result = asyncio.run(adapter.send("chat", "interim status", metadata=metadata))
    assert result.success and adapter.linear.creates == [] and adapter.linear.readbacks == []
    assert asyncio.run(adapter.send("chat", "The answer.", metadata={"notify": True})).success
    assert [c["body"] for c in adapter.linear.creates] == ["The answer."]


def test_silence_answers_a_mention_and_stays_silent_elsewhere(adapter):
    adapter.targets.update({"mention": destination(trigger="mention"), "assigned": destination(trigger="assigned"),
                            "log": {"deliver": "log", "deliver_extra": {}, "reply_id": ""}})
    for chat in adapter.targets:
        assert asyncio.run(adapter.send(chat, "[SILENT]", metadata={"notify": True})).success
    assert [c["body"] for c in adapter.linear.creates] == [webhook_linear.LINEAR_SILENCE_ACK]


def test_deliver_only_refuses_rather_than_post_unkeyed(adapter):
    result = asyncio.run(adapter._direct_deliver("The answer.", destination()))
    assert not result.success and adapter.linear.creates == []
