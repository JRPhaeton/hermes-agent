"""Telegram-native UX gates: inbound reaction replies, lifecycle emoji overrides, inline-button gate.

Covers ``telegram.reaction_replies`` (user reaction on a bot message → ordinary user turn),
``telegram.reaction_emojis`` (lifecycle ack overrides), ``telegram.buttons`` (inline keyboards on
clarify / exec-approval / slash-confirm prompts) and the config.yaml → extra/env bridge. Defaults
must reproduce the pre-gate behaviour: buttons on, reactions off, reaction replies off.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter, _apply_yaml_config

USER_ID = "4242"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("TELEGRAM_REACTIONS", "TELEGRAM_BUTTONS", "TELEGRAM_REACTION_REPLIES", "TELEGRAM_ALLOWED_USERS",
                "GATEWAY_ALLOW_ALL_USERS", "TELEGRAM_ALLOW_ALL_USERS"):
        monkeypatch.delenv(var, raising=False)


def _make_adapter(extra=None, *, allowed_users=USER_ID, monkeypatch=None):
    if monkeypatch is not None and allowed_users is not None:
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", allowed_users)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra=extra or {}))
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()
    adapter._app = MagicMock()
    adapter.handle_message = AsyncMock()
    return adapter


def _reaction_type(key):
    if key.startswith("custom:"):
        return SimpleNamespace(custom_emoji_id=key.split(":", 1)[1])
    return SimpleNamespace(emoji=key)


def _reaction_update(*, new, old=(), chat_id=777, chat_type="private", user_id=USER_ID, message_id=99,
                     update_id=5, user=None):
    if user is None:
        user = SimpleNamespace(id=int(user_id), username="john", full_name="John")
    message_reaction = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id, type=chat_type, is_forum=False), user=user, actor_chat=None,
        message_id=message_id, date=datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
        new_reaction=[_reaction_type(k) for k in new], old_reaction=[_reaction_type(k) for k in old])
    return SimpleNamespace(message_reaction=message_reaction, update_id=update_id, callback_query=None,
                           message=None, edited_message=None)


def _user_event(message_id="55", metadata=None):
    return MessageEvent(
        text="hello", message_type=MessageType.TEXT, message_id=message_id, metadata=metadata or {},
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="777", chat_type="dm", user_id=USER_ID))


# ── reaction_replies: config parsing ─────────────────────────────────────────


def test_reaction_replies_off_by_default(monkeypatch):
    adapter = _make_adapter(monkeypatch=monkeypatch)
    assert adapter._reaction_reply_map() == {}


@pytest.mark.parametrize("raw, expected", [
    (None, {}), (False, {}), ("", {}), ("false", {}), ("off", {}), ("0", {}),
    (True, {"👍": "yes", "👌": "yes", "👎": "no"}),
    ("true", {"👍": "yes", "👌": "yes", "👎": "no"}),
    ("default", {"👍": "yes", "👌": "yes", "👎": "no"}),
    ({"👍": "ship it", "custom:123": "ok", " 👎 ": " no ", "🔥": "", "🙏": None}, {"👍": "ship it", "custom:123": "ok", "👎": "no"}),
    ('{"👍": "yes please"}', {"👍": "yes please"}),
    ('"just a string"', {}), ("not json {", {}), (42, {}), (["👍"], {}),
])
def test_parse_reaction_reply_config(raw, expected):
    assert TelegramAdapter._parse_reaction_reply_config(raw) == expected


def test_env_beats_yaml_for_reaction_replies(monkeypatch):
    adapter = _make_adapter({"reaction_replies": True}, monkeypatch=monkeypatch)
    assert adapter._reaction_reply_map()["👍"] == "yes"
    monkeypatch.setenv("TELEGRAM_REACTION_REPLIES", "false")
    assert adapter._reaction_reply_map() == {}
    monkeypatch.setenv("TELEGRAM_REACTION_REPLIES", '{"🙏": "summarise"}')
    assert adapter._reaction_reply_map() == {"🙏": "summarise"}


# ── reaction_replies: added-reaction detection ───────────────────────────────


@pytest.mark.parametrize("new, old, expected", [
    (["👍"], [], ("👍", "yes")),
    (["👍"], ["👍"], None),              # unchanged (Telegram re-sent state)
    ([], ["👍"], None),                  # removal
    (["🔥"], [], None),                  # unmapped
    (["🔥", "👎"], ["🔥"], ("👎", "no")),  # second reaction added
    (["👎"], ["👍"], ("👎", "no")),      # swapped
    (["custom:555"], [], ("custom:555", "ship it")),
])
def test_added_reaction_reply_detection(monkeypatch, new, old, expected):
    adapter = _make_adapter(monkeypatch=monkeypatch)
    reply_map = {"👍": "yes", "👎": "no", "custom:555": "ship it"}
    update = _reaction_update(new=new, old=old)
    assert adapter._added_reaction_reply(update.message_reaction, reply_map) == expected


def test_reaction_keys_drop_paid_and_unknown_types(monkeypatch):
    adapter = _make_adapter(monkeypatch=monkeypatch)
    paid = SimpleNamespace(type="paid")
    assert adapter._reaction_keys([paid, _reaction_type("👍"), _reaction_type("custom:9")]) == ["👍", "custom:9"]
    assert adapter._reaction_keys(None) == []
    assert adapter._reaction_keys("👍") == []


# ── reaction_replies: handler → user turn ────────────────────────────────────


@pytest.mark.asyncio
async def test_disabled_feature_never_creates_turn(monkeypatch):
    adapter = _make_adapter(monkeypatch=monkeypatch)  # reaction_replies unset
    await adapter._handle_message_reaction(_reaction_update(new=["👍"]), None)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dm_reaction_becomes_framed_user_turn(monkeypatch):
    adapter = _make_adapter({"reaction_replies": True}, monkeypatch=monkeypatch)
    await adapter._handle_message_reaction(_reaction_update(new=["👍"], message_id=99, update_id=7), None)
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert isinstance(event, MessageEvent)
    assert event.message_type == MessageType.TEXT
    assert event.text == "[Reacted 👍 to an earlier message in this chat] yes"  # unknown message → explicit frame
    assert event.source.platform == Platform.TELEGRAM
    assert event.source.chat_id == "777" and event.source.chat_type == "dm" and event.source.user_id == USER_ID
    assert event.user_id == USER_ID
    assert event.message_id == "99" and event.reply_to_message_id == "99"
    assert event.reply_to_text is None and event.reply_to_is_own_message is False
    assert event.platform_update_id == 7
    assert event.metadata["telegram_reaction_reply"] is True
    assert event.metadata["telegram_reaction"] == "👍"
    assert event.metadata["telegram_reaction_target_message_id"] == "99"
    assert event.allow_gateway_control is True  # "/approve"-style map values dispatch as commands


@pytest.mark.asyncio
async def test_reaction_on_known_bot_message_quotes_it(monkeypatch):
    adapter = _make_adapter({"reaction_replies": {"👍": "yes, go ahead"}}, monkeypatch=monkeypatch)
    adapter._remember_own_message("777", 99, "Plan: migrate CT 336 tonight")
    await adapter._handle_message_reaction(_reaction_update(new=["👍"], message_id=99), None)
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "yes, go ahead"  # quote frame comes from the gateway's reply block
    assert event.reply_to_text == "Plan: migrate CT 336 tonight"
    assert event.reply_to_is_own_message is True


@pytest.mark.asyncio
async def test_group_reaction_requires_bot_message(monkeypatch):
    adapter = _make_adapter({"reaction_replies": True}, monkeypatch=monkeypatch)
    group = dict(chat_id=-100123, chat_type="supergroup", message_id=5)
    await adapter._handle_message_reaction(_reaction_update(new=["👍"], **group), None)
    adapter.handle_message.assert_not_awaited()  # a 👍 on a human's message is chatter
    adapter._remember_own_message("-100123", "5", "Shall I proceed?")
    await adapter._handle_message_reaction(_reaction_update(new=["👍"], **group), None)
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_type == "group" and event.reply_to_is_own_message is True
    assert event.text == "yes"


@pytest.mark.asyncio
async def test_removal_and_unmapped_reactions_never_create_turn(monkeypatch):
    adapter = _make_adapter({"reaction_replies": True}, monkeypatch=monkeypatch)
    await adapter._handle_message_reaction(_reaction_update(new=[], old=["👍"]), None)
    await adapter._handle_message_reaction(_reaction_update(new=["🔥"]), None)
    await adapter._handle_message_reaction(_reaction_update(new=["👍"], old=["👍"]), None)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_unauthorized_reactor_is_ignored(monkeypatch):
    adapter = _make_adapter({"reaction_replies": True}, allowed_users="1", monkeypatch=monkeypatch)
    await adapter._handle_message_reaction(_reaction_update(new=["👍"]), None)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_identity_fails_closed(monkeypatch):
    adapter = _make_adapter({"reaction_replies": True}, monkeypatch=monkeypatch)
    update = _reaction_update(new=["👍"], user=SimpleNamespace(id=None, username="", full_name=""))
    await adapter._handle_message_reaction(update, None)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_reaction_update_is_ignored(monkeypatch):
    adapter = _make_adapter({"reaction_replies": True}, monkeypatch=monkeypatch)
    await adapter._handle_message_reaction(SimpleNamespace(message_reaction=None), None)
    adapter.handle_message.assert_not_awaited()


# ── own-message index ────────────────────────────────────────────────────────


def test_own_message_index_is_bounded_lru(monkeypatch):
    adapter = _make_adapter(monkeypatch=monkeypatch)
    monkeypatch.setattr(TelegramAdapter, "_OWN_MESSAGE_INDEX_MAX", 3)
    for i in range(5):
        adapter._remember_own_message("777", i, f"msg {i}")
    assert list(adapter._own_message_index) == [("777", "2"), ("777", "3"), ("777", "4")]
    assert adapter._own_message_text("777", 1) == (False, None)
    assert adapter._own_message_text("777", "4") == (True, "msg 4")


def test_own_message_text_falls_back_to_rich_sent_store(monkeypatch):
    adapter = _make_adapter(monkeypatch=monkeypatch)
    from gateway import rich_sent_store
    monkeypatch.setattr(rich_sent_store, "lookup", lambda chat_id, message_id: "rich body" if message_id == "8" else None)
    assert adapter._own_message_text("777", 8) == (True, "rich body")
    assert adapter._own_message_text("777", 9) == (False, None)


def test_remember_own_message_never_raises(monkeypatch):
    adapter = _make_adapter(monkeypatch=monkeypatch)
    adapter._remember_own_message(None, None, "x")  # no-op
    adapter._remember_own_message("777", 1, None)
    assert adapter._own_message_text("777", 1) == (True, None)


# ── lifecycle reactions (send side) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_lifecycle_reactions_skip_reaction_reply_turns(monkeypatch):
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter(monkeypatch=monkeypatch)
    await adapter.on_processing_start(_user_event(metadata={"telegram_reaction_reply": True}))
    await adapter.on_processing_complete(_user_event(metadata={"telegram_reaction_reply": True}), ProcessingOutcome.SUCCESS)
    adapter._bot.set_message_reaction.assert_not_awaited()  # would land on the BOT's own message
    await adapter.on_processing_start(_user_event())
    adapter._bot.set_message_reaction.assert_awaited_once_with(chat_id=777, message_id=55, reaction="👀")


@pytest.mark.asyncio
async def test_reaction_emojis_override_and_null_phase(monkeypatch):
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter({"reaction_emojis": {"processing": "🫡", "success": None}}, monkeypatch=monkeypatch)
    await adapter.on_processing_start(_user_event())
    adapter._bot.set_message_reaction.assert_awaited_with(chat_id=777, message_id=55, reaction="🫡")
    await adapter.on_processing_complete(_user_event(), ProcessingOutcome.SUCCESS)
    adapter._bot.set_message_reaction.assert_awaited_with(chat_id=777, message_id=55, reaction=None)  # cleared
    await adapter.on_processing_complete(_user_event(), ProcessingOutcome.FAILURE)
    adapter._bot.set_message_reaction.assert_awaited_with(chat_id=777, message_id=55, reaction="👎")  # default kept


@pytest.mark.asyncio
async def test_lifecycle_defaults_unchanged(monkeypatch):
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter(monkeypatch=monkeypatch)
    await adapter.on_processing_complete(_user_event(), ProcessingOutcome.SUCCESS)
    adapter._bot.set_message_reaction.assert_awaited_with(chat_id=777, message_id=55, reaction="👍")
    await adapter.on_processing_complete(_user_event(), ProcessingOutcome.CANCELLED)
    adapter._bot.set_message_reaction.assert_awaited_with(chat_id=777, message_id=55, reaction=None)


def test_unsettable_reaction_emoji_warns_once(monkeypatch, caplog):
    constants = pytest.importorskip("telegram.constants")
    if not hasattr(constants, "ReactionEmoji"):
        pytest.skip("SDK does not expose ReactionEmoji")
    adapter = _make_adapter({"reaction_emojis": {"success": "✅"}}, monkeypatch=monkeypatch)
    with caplog.at_level("WARNING"):
        assert adapter._reaction_emojis()["success"] == "✅"
        adapter._reaction_emojis()
    assert sum("bot-settable" in rec.getMessage() for rec in caplog.records) == 1


# ── buttons gate ─────────────────────────────────────────────────────────────


def test_buttons_default_on_and_env_override(monkeypatch):
    adapter = _make_adapter(monkeypatch=monkeypatch)
    assert adapter._buttons_enabled() is True
    assert _make_adapter({"buttons": False}, monkeypatch=monkeypatch)._buttons_enabled() is False
    monkeypatch.setenv("TELEGRAM_BUTTONS", "false")
    assert _make_adapter({"buttons": True}, monkeypatch=monkeypatch)._buttons_enabled() is False


@pytest.mark.asyncio
async def test_clarify_keyboard_by_default(monkeypatch):
    adapter = _make_adapter(monkeypatch=monkeypatch)
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
    result = await adapter.send_clarify("777", "Which env?", ["prod", "staging"], "cl-1", "agent:main:telegram:dm:777")
    assert result.success and result.message_id == "42"
    kwargs = adapter._bot.send_message.call_args.kwargs
    assert kwargs["reply_markup"] is not None
    assert adapter._clarify_state["cl-1"] == "agent:main:telegram:dm:777"
    assert adapter._own_message_text("777", 42)[0] is True  # prompt indexed for reaction attribution


@pytest.mark.asyncio
async def test_clarify_text_fallback_when_buttons_off(monkeypatch):
    adapter = _make_adapter({"buttons": False}, monkeypatch=monkeypatch)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="43"))
    marked = []
    import tools.clarify_gateway as clarify_gateway
    monkeypatch.setattr(clarify_gateway, "mark_awaiting_text", lambda cid: marked.append(cid) or True)
    result = await adapter.send_clarify("777", "Which env?", ["prod", "staging"], "cl-2", "sk")
    assert result.success
    adapter._bot.send_message.assert_not_awaited()
    text = adapter.send.await_args.kwargs["content"]
    assert "1. prod" in text and "2. staging" in text and "Reply with the number" in text
    assert marked == ["cl-2"] and "cl-2" not in adapter._clarify_state


@pytest.mark.asyncio
async def test_exec_approval_text_fallback_when_buttons_off(monkeypatch):
    adapter = _make_adapter({"buttons": False}, monkeypatch=monkeypatch)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="44"))
    result = await adapter.send_exec_approval("777", "rm -rf /tmp/x", "sk", description="deletes files", allow_permanent=False)
    assert result.success
    adapter._bot.send_message.assert_not_awaited()
    args, kwargs = adapter.send.await_args
    text = args[1]
    assert "rm -rf /tmp/x" in text and "/approve" in text and "/deny" in text and "approve always" not in text
    assert kwargs["metadata"]["is_approval_prompt"] is True
    assert adapter._approval_state == {}


@pytest.mark.asyncio
async def test_exec_approval_keyboard_by_default(monkeypatch):
    adapter = _make_adapter(monkeypatch=monkeypatch)
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=45))
    result = await adapter.send_exec_approval("777", "rm -rf /tmp/x", "sk")
    assert result.success and adapter._bot.send_message.call_args.kwargs["reply_markup"] is not None


@pytest.mark.asyncio
async def test_slash_confirm_declines_when_buttons_off(monkeypatch):
    adapter = _make_adapter({"buttons": False}, monkeypatch=monkeypatch)
    result = await adapter.send_slash_confirm("777", "Reload", "Reload MCP?", "sk", "sc-1")
    assert result.success is False and result.error == "Not supported"  # gateway text fallback by contract
    adapter._bot.send_message.assert_not_awaited()


# ── handler registration + yaml bridge ───────────────────────────────────────


def test_register_handlers_adds_reaction_handler_in_own_group(monkeypatch):
    adapter = _make_adapter(monkeypatch=monkeypatch)
    app = MagicMock()
    adapter._register_handlers(app)
    groups = [call.kwargs.get("group") for call in app.add_handler.call_args_list]
    assert groups.count(1) == 1 and 99 in groups  # reaction handler never shares group 0 with message handlers


def test_apply_yaml_config_bridges_ux_gates(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BUTTONS", "")
    extras = _apply_yaml_config({}, {
        "buttons": False, "reaction_replies": {"👍": "yes"}, "reaction_emojis": {"processing": "🫡"}})
    assert extras["buttons"] is False
    assert extras["reaction_replies"] == {"👍": "yes"}
    assert extras["reaction_emojis"] == {"processing": "🫡"}
    import os
    assert os.getenv("TELEGRAM_BUTTONS") == "false"


def test_apply_yaml_config_legacy_block_unchanged(monkeypatch):
    assert _apply_yaml_config({}, {}) is None
