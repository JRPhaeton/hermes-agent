"""Telegram reaction replies, the inline-button gate, and late clarify taps.

* ``reaction_replies`` / ``TELEGRAM_REACTION_REPLIES`` parsing (off by default, ``true`` = default map,
  custom mapping as YAML dict / JSON / ``👍=yes`` pairs) and the YAML→env bridge.
* An opted-in reaction on one of the bot's own messages becomes a normal user turn carrying the mapped
  reply and the bot message as reply context; removals, unmapped emoji, bot actors, unknown messages and
  unauthorized reactors never start a turn.
* ``buttons_enabled`` / ``TELEGRAM_INLINE_BUTTONS`` (default on) falls back to the legacy text prompts.
* A clarify button tapped after the prompt expired is routed as a user turn instead of "please /retry".
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from plugins.platforms.telegram.adapter import TelegramAdapter, _apply_yaml_config


# ---------------------------------------------------------------------------
# Fixtures / stand-ins
# ---------------------------------------------------------------------------

BOT_ID = 999_999


def _make_adapter(extra=None, *, authorized: bool = True) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra=extra or {}))
    adapter._bot = AsyncMock()
    adapter._bot.id = BOT_ID
    adapter._app = MagicMock()
    adapter._is_callback_user_authorized = lambda *a, **k: authorized
    adapter.handle_message = AsyncMock()
    return adapter


def _chat(chat_id=123, chat_type="private", is_forum=False):
    return SimpleNamespace(id=chat_id, type=chat_type, title=None if chat_type == "private" else "Ops", full_name="John Doe", is_forum=is_forum)


def _user(user_id=777, is_bot=False):
    return SimpleNamespace(id=user_id, is_bot=is_bot, username="john", full_name="John Doe", first_name="John")


def _reaction_update(new, *, old=(), user=None, chat=None, message_id=100, actor_chat=None):
    mr = SimpleNamespace(
        chat=chat or _chat(), user=user if user is not None else (_user() if actor_chat is None else None), actor_chat=actor_chat,
        message_id=message_id, new_reaction=[SimpleNamespace(emoji=e, custom_emoji_id=None) for e in new],
        old_reaction=[SimpleNamespace(emoji=e, custom_emoji_id=None) for e in old], date=None)
    return SimpleNamespace(message_reaction=mr, callback_query=None)


def _clear_clarify_state():
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


_ENV_KEYS = ("TELEGRAM_REACTION_REPLIES", "TELEGRAM_INLINE_BUTTONS", "TELEGRAM_REACTIONS")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Start every test with the three gates unset and leave them unset: the YAML bridge writes
    ``os.environ`` directly, and ``monkeypatch.delenv`` on an absent key records nothing to restore."""
    import os
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    _clear_clarify_state()
    yield
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    _clear_clarify_state()


# ---------------------------------------------------------------------------
# Map parsing + config bridge
# ---------------------------------------------------------------------------

class TestReactionReplyMap:
    def test_off_by_default(self):
        assert _make_adapter()._reaction_reply_map() is None

    @pytest.mark.parametrize("raw", [True, "true", "1", "on", "all", "default"])
    def test_true_means_default_map(self, raw):
        assert TelegramAdapter._parse_reaction_reply_map(raw) == {"👍": "yes", "👌": "yes", "👎": "no"}

    @pytest.mark.parametrize("raw", [False, "false", "0", "off", "", None, "none"])
    def test_falsey_disables(self, raw):
        assert TelegramAdapter._parse_reaction_reply_map(raw) is None

    def test_dict_map_normalizes_variation_selectors_and_drops_empty(self):
        parsed = TelegramAdapter._parse_reaction_reply_map({"❤️": "love it", "👍": " yes ", "🔥": "", "👎": None})
        assert parsed == {"❤": "love it", "👍": "yes"}

    def test_json_and_pairs_text_forms(self):
        assert TelegramAdapter._parse_reaction_reply_map('{"👍": "yes", "🤔": "explain"}') == {"👍": "yes", "🤔": "explain"}
        assert TelegramAdapter._parse_reaction_reply_map("👍=yes, 👎=no; 🤔: tell me more") == {"👍": "yes", "👎": "no", "🤔": "tell me more"}

    def test_invalid_json_disables(self):
        assert TelegramAdapter._parse_reaction_reply_map("{not json") is None

    def test_env_drives_adapter(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REACTION_REPLIES", "👍=/approve")
        assert _make_adapter()._reaction_reply_map() == {"👍": "/approve"}


class TestYamlBridge:
    def test_bridges_new_keys_to_env(self, monkeypatch):
        _apply_yaml_config({}, {"reaction_replies": {"👍": "yes", "👎": "no"}, "buttons_enabled": False, "reactions_enabled": True})
        import json, os
        assert json.loads(os.environ["TELEGRAM_REACTION_REPLIES"]) == {"👍": "yes", "👎": "no"}
        assert os.environ["TELEGRAM_INLINE_BUTTONS"] == "false"
        assert os.environ["TELEGRAM_REACTIONS"] == "true"

    def test_bool_reaction_replies_and_legacy_reactions_key_wins(self, monkeypatch):
        _apply_yaml_config({}, {"reaction_replies": True, "reactions": False, "reactions_enabled": True})
        import os
        assert os.environ["TELEGRAM_REACTION_REPLIES"] == "true"
        assert os.environ["TELEGRAM_REACTIONS"] == "false"  # explicit `reactions` beats the alias

    def test_absent_keys_leave_env_alone(self):
        import os
        _apply_yaml_config({}, {})
        assert "TELEGRAM_REACTION_REPLIES" not in os.environ
        assert "TELEGRAM_INLINE_BUTTONS" not in os.environ


# ---------------------------------------------------------------------------
# Sent-text index (what the bot said, keyed by message id)
# ---------------------------------------------------------------------------

class TestSentTextIndex:
    @pytest.mark.asyncio
    async def test_legacy_send_is_indexed_with_thread(self):
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=77))
        result = await adapter.send("123", "Plan: ship it?", metadata={"thread_id": "42"})
        assert result.success and result.message_id == "77"
        assert adapter._lookup_sent_text("123", "77") == {"text": "Plan: ship it?", "thread_id": "42"}

    @pytest.mark.asyncio
    async def test_control_prompt_is_indexed(self):
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=5))
        await adapter.send_clarify("123", "Deploy?", ["yes", "no"], "cl1", "agent:main:telegram:dm:123")
        assert "Deploy?" in adapter._lookup_sent_text("123", "5")["text"]

    @pytest.mark.asyncio
    async def test_finalized_edit_is_indexed(self):
        adapter = _make_adapter()
        adapter._edit_message_impl = AsyncMock(return_value=SendResult(success=True, message_id="9"))
        await adapter.edit_message("123", "9", "final text", finalize=True)
        assert adapter._lookup_sent_text("123", "9")["text"] == "final text"
        await adapter.edit_message("123", "10", "draft", finalize=False)
        assert adapter._lookup_sent_text("123", "10") is None

    def test_index_is_bounded_lru(self):
        adapter = _make_adapter()
        adapter._SENT_TEXT_INDEX_MAX = 3
        for mid in range(5):
            adapter._remember_sent_text("c", mid, f"t{mid}")
        assert adapter._lookup_sent_text("c", 0) is None and adapter._lookup_sent_text("c", 4)["text"] == "t4"
        assert len(adapter._sent_text_index()) == 3

    def test_falls_back_to_rich_sent_store(self, monkeypatch):
        from gateway import rich_sent_store
        monkeypatch.setattr(rich_sent_store, "lookup", lambda c, m: "rich final" if (c, m) == ("1", "2") else None)
        adapter = _make_adapter()
        assert adapter._lookup_sent_text("1", "2") == {"text": "rich final", "thread_id": None}
        assert adapter._lookup_sent_text("1", "3") is None


# ---------------------------------------------------------------------------
# Inbound reaction → turn
# ---------------------------------------------------------------------------

def _enabled_adapter(monkeypatch, raw="true", **kw):
    monkeypatch.setenv("TELEGRAM_REACTION_REPLIES", raw)
    adapter = _make_adapter(**kw)
    adapter._remember_sent_text("123", 100, "Proposed plan: migrate tonight. OK?", None)
    return adapter


class TestReactionReplies:
    @pytest.mark.asyncio
    async def test_thumbs_up_on_bot_message_becomes_yes_turn(self, monkeypatch):
        adapter = _enabled_adapter(monkeypatch)
        await adapter._on_platform_update(_reaction_update(["👍"]), None)
        adapter.handle_message.assert_awaited_once()
        event: MessageEvent = adapter.handle_message.await_args.args[0]
        assert event.text == "[reaction 👍] yes"
        assert event.message_type is MessageType.TEXT
        assert event.reply_to_message_id == "100" and event.reply_to_is_own_message is True
        assert event.reply_to_text == "Proposed plan: migrate tonight. OK?"
        assert event.source.chat_id == "123" and event.source.chat_type == "dm" and event.source.user_id == "777"
        assert event.metadata["telegram_reaction_turn"] is True
        assert event.metadata["telegram_reaction"] == {"emoji": "👍", "reply": "yes", "message_id": "100"}

    @pytest.mark.asyncio
    async def test_thumbs_down_is_no(self, monkeypatch):
        adapter = _enabled_adapter(monkeypatch)
        await adapter._on_platform_update(_reaction_update(["👎"]), None)
        assert adapter.handle_message.await_args.args[0].text == "[reaction 👎] no"

    @pytest.mark.asyncio
    async def test_slash_reply_is_dispatched_verbatim_as_command(self, monkeypatch):
        adapter = _enabled_adapter(monkeypatch, raw="👍=/approve")
        await adapter._on_platform_update(_reaction_update(["👍"]), None)
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/approve" and event.get_command() == "approve"

    @pytest.mark.asyncio
    async def test_disabled_by_default_even_on_bot_message(self):
        adapter = _make_adapter()
        adapter._remember_sent_text("123", 100, "bot said", None)
        await adapter._on_platform_update(_reaction_update(["👍"]), None)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unmapped_emoji_starts_no_turn(self, monkeypatch):
        adapter = _enabled_adapter(monkeypatch)
        await adapter._on_platform_update(_reaction_update(["🔥"]), None)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_removal_starts_no_turn_but_swap_does(self, monkeypatch):
        adapter = _enabled_adapter(monkeypatch)
        await adapter._on_platform_update(_reaction_update([], old=["👍"]), None)
        adapter.handle_message.assert_not_awaited()
        await adapter._on_platform_update(_reaction_update(["👎"], old=["👍"]), None)
        assert adapter.handle_message.await_args.args[0].text == "[reaction 👎] no"

    @pytest.mark.asyncio
    async def test_reaction_on_unknown_message_is_ignored(self, monkeypatch):
        adapter = _enabled_adapter(monkeypatch)
        await adapter._on_platform_update(_reaction_update(["👍"], message_id=555), None)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_bot_actors_are_ignored(self, monkeypatch):
        adapter = _enabled_adapter(monkeypatch)
        await adapter._on_platform_update(_reaction_update(["👍"], user=_user(user_id=424242, is_bot=True)), None)
        await adapter._on_platform_update(_reaction_update(["👍"], user=_user(user_id=BOT_ID)), None)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unauthorized_reactor_is_ignored(self, monkeypatch):
        adapter = _enabled_adapter(monkeypatch, authorized=False)
        await adapter._on_platform_update(_reaction_update(["👍"]), None)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_custom_emoji_only_is_ignored(self, monkeypatch):
        adapter = _enabled_adapter(monkeypatch)
        update = _reaction_update([])
        update.message_reaction.new_reaction = [SimpleNamespace(emoji=None, custom_emoji_id="5368324170671202286")]
        await adapter._on_platform_update(update, None)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_forum_topic_routes_to_the_topic_session(self, monkeypatch):
        adapter = _enabled_adapter(monkeypatch)
        adapter._remember_sent_text("-1001", 7, "topic reply", "42")
        chat = _chat(chat_id=-1001, chat_type="supergroup", is_forum=True)
        await adapter._on_platform_update(_reaction_update(["👍"], chat=chat, message_id=7), None)
        source = adapter.handle_message.await_args.args[0].source
        assert source.chat_type == "group" and source.thread_id == "42" and source.chat_id == "-1001"

    @pytest.mark.asyncio
    async def test_non_reaction_updates_are_ignored(self, monkeypatch):
        adapter = _enabled_adapter(monkeypatch)
        await adapter._on_platform_update(SimpleNamespace(message_reaction=None, message=object()), None)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handler_errors_never_escape_into_ptb(self, monkeypatch):
        adapter = _enabled_adapter(monkeypatch)
        adapter.handle_message = AsyncMock(side_effect=RuntimeError("boom"))
        await adapter._on_platform_update(_reaction_update(["👍"]), None)  # no raise

    @pytest.mark.asyncio
    async def test_rides_the_group_99_observer_without_a_plugin_subscriber(self, monkeypatch):
        """Reaction replies share the single catch-all registration (#64176) and must not depend on a
        gateway_platform_event subscriber being present."""
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: False)
        adapter = _enabled_adapter(monkeypatch)
        adapter._platform_event_handler = None
        await adapter._on_platform_update(_reaction_update(["👍"]), None)
        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_observer_hook_still_fires_after_a_reaction_turn(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
        adapter = _enabled_adapter(monkeypatch)
        seen = []

        async def _hook(event, source):
            seen.append((event["event_type"], source.user_id))
        adapter._platform_event_handler = _hook
        await adapter._on_platform_update(_reaction_update(["👍"]), None)
        adapter.handle_message.assert_awaited_once()
        assert seen == [("reaction", "777")]


class TestLifecycleReactionsOnReactionTurns:
    @pytest.mark.asyncio
    async def test_lifecycle_reactions_skip_reaction_turns(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
        adapter = _make_adapter()
        event = MessageEvent(text="[reaction 👍] yes", message_type=MessageType.TEXT,
                             source=SimpleNamespace(chat_id="123"), message_id="100", metadata={"telegram_reaction_turn": True})
        await adapter.on_processing_start(event)
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        adapter._bot.set_message_reaction.assert_not_awaited()
        plain = MessageEvent(text="hi", message_type=MessageType.TEXT, source=SimpleNamespace(chat_id="123"), message_id="101")
        await adapter.on_processing_start(plain)
        adapter._bot.set_message_reaction.assert_awaited_once_with(chat_id=123, message_id=101, reaction="\U0001f440")


# ---------------------------------------------------------------------------
# Inline-button gate
# ---------------------------------------------------------------------------

class TestButtonsGate:
    def test_buttons_on_by_default(self):
        assert TelegramAdapter._buttons_enabled() is True

    @pytest.mark.asyncio
    async def test_clarify_falls_back_to_numbered_text(self, monkeypatch):
        from tools import clarify_gateway as cg
        monkeypatch.setenv("TELEGRAM_INLINE_BUTTONS", "false")
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
        entry = cg.register("cl9", "agent:main:telegram:dm:123", "Deploy?", ["yes", "no"])
        result = await adapter.send_clarify("123", "Deploy?", ["yes", "no"], "cl9", "agent:main:telegram:dm:123")
        assert result.success
        kwargs = adapter._bot.send_message.call_args.kwargs
        assert kwargs.get("reply_markup") is None
        assert "1\\. yes" in kwargs["text"] or "1. yes" in kwargs["text"]
        assert entry.awaiting_text is True  # gateway text-intercept armed, legacy behaviour

    @pytest.mark.asyncio
    async def test_exec_approval_and_slash_confirm_defer_to_text_fallback(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_INLINE_BUTTONS", "0")
        adapter = _make_adapter()
        ea = await adapter.send_exec_approval("123", "rm -rf /tmp/x", "agent:main:telegram:dm:123")
        sc = await adapter.send_slash_confirm("123", "t", "m", "agent:main:telegram:dm:123", "c1")
        assert ea.success is False and sc.success is False
        adapter._bot.send_message.assert_not_called()
        assert adapter._approval_state == {} and adapter._slash_confirm_state == {}

    @pytest.mark.asyncio
    async def test_buttons_enabled_keeps_inline_keyboard(self):
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
        await adapter.send_exec_approval("123", "rm -rf /tmp/x", "agent:main:telegram:dm:123")
        assert adapter._bot.send_message.call_args.kwargs["reply_markup"] is not None


# ---------------------------------------------------------------------------
# Late clarify taps
# ---------------------------------------------------------------------------

def _callback_update(data, *, user_id=777, chat=None, message_id=100, text="❓ Deploy?\n\n1. yes\n2. no"):
    chat = chat or _chat()
    message = SimpleNamespace(chat=chat, chat_id=chat.id, message_id=message_id, text=text, message_thread_id=None,
                              is_topic_message=False, reply_to_message=None)
    query = SimpleNamespace(data=data, message=message, from_user=_user(user_id=user_id), answer=AsyncMock(), edit_message_text=AsyncMock())
    return SimpleNamespace(callback_query=query, message_reaction=None)


class TestLateClarifyTap:
    @pytest.mark.asyncio
    async def test_expired_prompt_tap_is_routed_as_user_turn(self):
        adapter = _make_adapter()
        update = _callback_update("cl:gone:1")  # no entry, no adapter state: timeout or restart
        await adapter._handle_callback_query(update, None)
        query = update.callback_query
        query.answer.assert_awaited()
        assert "✓ no" in query.answer.await_args.kwargs["text"]
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "no" and event.reply_to_is_own_message and event.reply_to_message_id == "100"
        assert event.metadata == {"telegram_late_clarify_answer": True}
        assert "expired" in query.edit_message_text.await_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_second_tap_on_same_expired_prompt_is_not_duplicated(self):
        adapter = _make_adapter()
        await adapter._handle_callback_query(_callback_update("cl:gone:1"), None)
        update = _callback_update("cl:gone:0")
        await adapter._handle_callback_query(update, None)
        adapter.handle_message.assert_awaited_once()
        assert update.callback_query.answer.await_args.kwargs["text"] == "This prompt has already been resolved."

    @pytest.mark.asyncio
    async def test_live_prompt_tap_still_resolves_the_tool_not_a_turn(self):
        from tools import clarify_gateway as cg
        adapter = _make_adapter()
        adapter._clarify_state["live"] = "agent:main:telegram:dm:123"
        cg.register("live", "agent:main:telegram:dm:123", "Deploy?", ["yes", "no"])
        waiter = asyncio.get_running_loop().run_in_executor(None, cg.wait_for_response, "live", 5.0)
        await asyncio.sleep(0.05)
        await adapter._handle_callback_query(_callback_update("cl:live:0"), None)
        assert await asyncio.wait_for(waiter, 5.0) == "yes"
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unauthorized_tap_is_refused(self):
        adapter = _make_adapter(authorized=False)
        update = _callback_update("cl:gone:1")
        await adapter._handle_callback_query(update, None)
        adapter.handle_message.assert_not_awaited()
        assert "not authorized" in update.callback_query.answer.await_args.kwargs["text"]

    def test_choice_text_prefers_entry_then_prompt_body(self):
        from tools import clarify_gateway as cg
        cg.register("x", "s", "q", ["alpha", "beta"])
        assert TelegramAdapter._clarify_choice_text("x", 1, None) == "beta"
        assert TelegramAdapter._clarify_choice_text("missing", 1, "❓ q\n\n1. one\n2. two") == "two"
        assert TelegramAdapter._clarify_choice_text("missing", 4, "❓ q\n\n1. one") == "choice 5"
