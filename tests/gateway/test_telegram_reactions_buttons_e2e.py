"""End-to-end (mocked Telegram Bot API): clarify → inline buttons → callback round-trip into the pending
clarify entry, then a 👍 reaction on that same bot message → a framed user turn, plus lifecycle acks.

The fake API records every ``send_message`` / ``set_message_reaction`` call; the adapter is the real
``TelegramAdapter`` with ``reaction_replies`` enabled and the default (on) button gate.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource
from plugins.platforms.telegram import adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import clarify_gateway

USER_ID = 4242
CHAT_ID = 777
SESSION_KEY = f"agent:main:telegram:dm:{CHAT_ID}"


class FakeTelegramAPI:
    """Minimal Bot API double: assigns message ids, records sends and reactions."""

    def __init__(self):
        self.sent = []
        self.reactions = []
        self._next_id = 100

    async def send_message(self, **kwargs):
        self._next_id += 1
        self.sent.append(kwargs)
        return SimpleNamespace(message_id=self._next_id, chat_id=kwargs.get("chat_id"), text=kwargs.get("text"))

    async def set_message_reaction(self, chat_id, message_id, reaction=None, **_):
        self.reactions.append((chat_id, message_id, reaction))
        return True


@pytest.fixture
def api_and_adapter(monkeypatch):
    for var in ("TELEGRAM_REACTIONS", "TELEGRAM_BUTTONS", "TELEGRAM_REACTION_REPLIES", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", str(USER_ID))
    # Capture keyboards as plain data so the assertions hold with the real SDK and the test mock alike.
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", lambda text, callback_data: (text, callback_data))
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", lambda rows: {"rows": rows})
    api = FakeTelegramAPI()
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={"reaction_replies": True}))
    adapter._bot = api
    adapter._app = MagicMock()
    yield api, adapter
    for cid in list(clarify_gateway._entries):
        clarify_gateway._entries.pop(cid, None)
    clarify_gateway._session_index.clear()


def _callback_update(data, *, message_id, text, user_id=USER_ID):
    query = SimpleNamespace(
        data=data, from_user=SimpleNamespace(id=user_id, first_name="John"),
        message=SimpleNamespace(chat_id=CHAT_ID, chat=SimpleNamespace(type="private"), message_thread_id=None,
                                message_id=message_id, text=text),
        answer=AsyncMock(), edit_message_text=AsyncMock())
    return SimpleNamespace(callback_query=query), query


def _reaction_update(emoji, *, message_id, old=(), user_id=USER_ID, update_id=11):
    message_reaction = SimpleNamespace(
        chat=SimpleNamespace(id=CHAT_ID, type="private", is_forum=False),
        user=SimpleNamespace(id=user_id, username="john", full_name="John"), actor_chat=None,
        message_id=message_id, date=datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
        new_reaction=[SimpleNamespace(emoji=emoji)] if emoji else [],
        old_reaction=[SimpleNamespace(emoji=e) for e in old])
    return SimpleNamespace(message_reaction=message_reaction, update_id=update_id, callback_query=None,
                           message=None, edited_message=None)


@pytest.mark.asyncio
async def test_clarify_button_roundtrip_then_reaction_reply(api_and_adapter):
    api, adapter = api_and_adapter
    choices = ["Yes, deploy", "No, wait"]
    clarify_gateway.register("cid-1", SESSION_KEY, "Deploy now?", choices)

    # 1. The clarify prompt renders as an inline keyboard: one numbered button per choice + "Other".
    result = await adapter.send_clarify(str(CHAT_ID), "Deploy now?", choices, "cid-1", SESSION_KEY)
    assert result.success and result.message_id == "101"
    prompt = api.sent[-1]
    assert prompt["chat_id"] == CHAT_ID and "Deploy now?" in prompt["text"]
    assert "1. Yes, deploy" in prompt["text"] and "2. No, wait" in prompt["text"]
    callback_data = [button[1] for row in prompt["reply_markup"]["rows"] for button in row]
    assert callback_data == ["cl:cid-1:0", "cl:cid-1:1", "cl:cid-1:other"]
    assert adapter._clarify_state["cid-1"] == SESSION_KEY

    # 2. Tapping button 2 resolves the pending clarify with the choice text (what the agent thread receives).
    update, query = _callback_update("cl:cid-1:1", message_id=101, text=prompt["text"])
    await adapter._handle_callback_query(update, None)
    assert clarify_gateway.wait_for_response("cid-1", timeout=1) == "No, wait"
    query.answer.assert_awaited_once()
    assert query.answer.await_args.kwargs["text"].startswith("✓")
    query.edit_message_text.assert_awaited_once()
    assert query.edit_message_text.await_args.kwargs["reply_markup"] is None  # keyboard removed
    assert "cid-1" not in adapter._clarify_state

    # 3. A second tap on the same (now resolved) prompt is answered, not re-applied.
    update2, query2 = _callback_update("cl:cid-1:0", message_id=101, text=prompt["text"])
    await adapter._handle_callback_query(update2, None)
    query2.answer.assert_awaited_once_with(text="This prompt has already been resolved.")

    # 4. John reacts 👍 to that bot message → a normal user turn quoting the prompt, mapped to "yes".
    adapter.handle_message = AsyncMock()
    await adapter._handle_message_reaction(_reaction_update("👍", message_id=101), None)
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert isinstance(event, MessageEvent) and event.message_type == MessageType.TEXT
    assert event.text == "yes"
    assert event.reply_to_message_id == "101" and event.reply_to_is_own_message is True
    assert "Deploy now?" in event.reply_to_text
    assert event.source.user_id == str(USER_ID) and event.source.chat_id == str(CHAT_ID) and event.source.chat_type == "dm"
    assert event.metadata == {"telegram_reaction_reply": True, "telegram_reaction": "👍",
                              "telegram_reaction_target_message_id": "101"}

    # 5. Unmapped emoji and removals stay silent; a stranger's 👍 is ignored (fail-closed auth).
    await adapter._handle_message_reaction(_reaction_update("🔥", message_id=101), None)
    await adapter._handle_message_reaction(_reaction_update(None, message_id=101, old=("👍",)), None)
    await adapter._handle_message_reaction(_reaction_update("👍", message_id=101, user_id=1), None)
    assert adapter.handle_message.await_count == 1

    # 6. 👎 maps to "no" through the same path.
    await adapter._handle_message_reaction(_reaction_update("👎", message_id=101, old=("👍",)), None)
    assert adapter.handle_message.await_args.args[0].text == "no"
    assert adapter.handle_message.await_count == 2


@pytest.mark.asyncio
async def test_unauthorized_button_tap_does_not_resolve(api_and_adapter):
    api, adapter = api_and_adapter
    clarify_gateway.register("cid-2", SESSION_KEY, "Proceed?", ["Yes", "No"])
    await adapter.send_clarify(str(CHAT_ID), "Proceed?", ["Yes", "No"], "cid-2", SESSION_KEY)
    update, query = _callback_update("cl:cid-2:0", message_id=api._next_id, text="❓ Proceed?", user_id=1)
    await adapter._handle_callback_query(update, None)
    query.answer.assert_awaited_once()
    assert query.answer.await_args.kwargs["text"].startswith("⛔")
    assert clarify_gateway._entries["cid-2"].event.is_set() is False
    assert adapter._clarify_state["cid-2"] == SESSION_KEY  # still armed for the real user


@pytest.mark.asyncio
async def test_lifecycle_acks_on_user_message_not_on_reaction_turns(api_and_adapter, monkeypatch):
    api, adapter = api_and_adapter
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    user_event = MessageEvent(
        text="deploy please", message_type=MessageType.TEXT, message_id="55",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id=str(CHAT_ID), chat_type="dm", user_id=str(USER_ID)))
    await adapter.on_processing_start(user_event)
    await adapter.on_processing_complete(user_event, ProcessingOutcome.SUCCESS)
    assert api.reactions == [(CHAT_ID, 55, "👀"), (CHAT_ID, 55, "👍")]

    reaction_turn = MessageEvent(
        text="yes", message_type=MessageType.TEXT, message_id="101", source=user_event.source,
        metadata={"telegram_reaction_reply": True, "telegram_reaction": "👍", "telegram_reaction_target_message_id": "101"})
    await adapter.on_processing_start(reaction_turn)
    await adapter.on_processing_complete(reaction_turn, ProcessingOutcome.SUCCESS)
    assert len(api.reactions) == 2  # never decorates the bot's own message
