"""End-to-end: a real ``TelegramAdapter`` over a mocked Telegram Bot API.

Two round-trips the unit tests only exercise piecewise:

1. **Inline button round-trip** — ``send_clarify`` renders ``cl:<id>:<idx>`` buttons; the pressed
   button's callback query is answered (``answerCallbackQuery``), the prompt is edited with the
   keyboard removed, and the choice text unblocks the agent thread waiting in
   ``clarify_gateway.wait_for_response`` — no extra agent turn. A tap after the prompt expired is
   delivered as a normal user turn instead.
2. **Reaction map** — a 👍 on a message the bot sent arrives at the runner as a ``yes`` turn that
   replies to that message, the agent's answer goes back out through the same API, and the
   lifecycle 👀/👍 reactions never land on the bot's own message. 🔥 (unmapped) and an
   unauthorized reactor start nothing.

The runner side is the real ``BasePlatformAdapter.handle_message`` path with a plain-function
message handler (no ``__self__`` — the adapter's callback auth then falls through to the
``TELEGRAM_ALLOWED_USERS`` allowlist exactly as in a single-profile deployment).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
import plugins.platforms.telegram.adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import clarify_gateway as cg

OWNER = 777
STRANGER = 888
BOT_ID = 999_999
CHAT = 123


class FakeBotAPI:
    """Records every Bot API method call; ``sendMessage`` hands back message objects with fresh ids."""

    def __init__(self):
        self.id = BOT_ID
        self.username = "hermes_e2e_bot"
        self.calls: list[tuple[str, dict]] = []
        self._next_id = 100

    def record(self, method: str, kwargs: dict) -> None:
        self.calls.append((method, dict(kwargs)))

    def of(self, method: str) -> list[dict]:
        return [kw for m, kw in self.calls if m == method]

    async def send_message(self, **kwargs):
        self.record("sendMessage", kwargs)
        message_id, self._next_id = self._next_id, self._next_id + 1
        return SimpleNamespace(
            message_id=message_id, chat_id=kwargs.get("chat_id"), text=kwargs.get("text"),
            message_thread_id=kwargs.get("message_thread_id"))

    async def set_message_reaction(self, **kwargs):
        self.record("setMessageReaction", kwargs)
        return True

    async def edit_message_text(self, **kwargs):
        self.record("editMessageText", kwargs)
        return True

    async def send_chat_action(self, **kwargs):
        self.record("sendChatAction", kwargs)
        return True

    async def get_me(self):
        return SimpleNamespace(id=self.id, username=self.username)

    def __getattr__(self, name):  # any other Bot API method: record and succeed
        async def _call(*args, **kwargs):
            self.record(name, kwargs)
            return True
        return _call


class _Button:
    """``InlineKeyboardButton`` stand-in (the gateway conftest mocks the SDK; keep the wire shape honest)."""

    def __init__(self, text, callback_data=None, **_):
        assert len(str(callback_data).encode("utf-8")) <= 64, "Telegram caps callback_data at 64 bytes"
        self.text, self.callback_data = text, callback_data


class _Keyboard:
    def __init__(self, rows):
        self.inline_keyboard = tuple(tuple(row) for row in rows)


class FakeCallbackQuery:
    """The pressed-button side of the API: ``answer`` is ``answerCallbackQuery``."""

    def __init__(self, api: FakeBotAPI, data: str, message, from_user):
        self._api, self.data, self.message, self.from_user = api, data, message, from_user

    async def answer(self, text=None, **kwargs):
        self._api.record("answerCallbackQuery", {"text": text, **kwargs})
        return True

    async def edit_message_text(self, **kwargs):
        self._api.record("editMessageText", kwargs)
        return True


def _chat():
    return SimpleNamespace(id=CHAT, type="private", title=None, full_name="John Doe", is_forum=False)


def _user(user_id=OWNER, is_bot=False):
    return SimpleNamespace(id=user_id, is_bot=is_bot, username="john", full_name="John Doe", first_name="John")


def _bot_message(message_id: int, text: str):
    return SimpleNamespace(chat=_chat(), chat_id=CHAT, message_id=message_id, text=text, message_thread_id=None,
                           is_topic_message=False, reply_to_message=None, from_user=_user(BOT_ID, is_bot=True))


def _callback_update(api, data, *, message, user_id=OWNER):
    return SimpleNamespace(callback_query=FakeCallbackQuery(api, data, message, _user(user_id)), message_reaction=None)


def _reaction_update(emojis, *, message_id: int, user_id=OWNER, old=()):
    mr = SimpleNamespace(
        chat=_chat(), user=_user(user_id), actor_chat=None, message_id=message_id, date=None,
        new_reaction=[SimpleNamespace(emoji=e, custom_emoji_id=None) for e in emojis],
        old_reaction=[SimpleNamespace(emoji=e, custom_emoji_id=None) for e in old])
    return SimpleNamespace(message_reaction=mr, callback_query=None)


def _clear_clarify_state():
    with cg._lock:
        cg._entries.clear()
        cg._session_index.clear()
        cg._notify_cbs.clear()


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", str(OWNER))
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.setenv("TELEGRAM_REACTION_REPLIES", "true")
    monkeypatch.delenv("TELEGRAM_INLINE_BUTTONS", raising=False)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: False)
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _Button)
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _Keyboard)
    _clear_clarify_state()
    api = FakeBotAPI()
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = api
    adapter._app = MagicMock()
    seen: list = []

    async def runner_handle_message(event):  # plain function on purpose: no __self__ → env allowlist auth
        seen.append(event)
        return f"On it: {event.text}"

    adapter.set_message_handler(runner_handle_message)
    yield SimpleNamespace(adapter=adapter, api=api, seen=seen)
    _clear_clarify_state()


async def _drain(adapter: TelegramAdapter) -> None:
    """Let the background turn spawned by handle_message run to completion."""
    for _ in range(50):
        tasks = [t for t in adapter._background_tasks if not t.done()]
        if not tasks:
            return
        await asyncio.wait(tasks, timeout=0.2)
    raise AssertionError("background turn did not finish")


# ---------------------------------------------------------------------------
# 1. Inline button round-trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clarify_button_round_trip_answers_callback_and_resolves_tool(rig):
    adapter, api = rig.adapter, rig.api
    session_key = "agent:main:telegram:dm:123"
    cg.register("cl-e2e", session_key, "Deploy to prod?", ["yes, deploy", "no, hold"])

    result = await adapter.send_clarify(str(CHAT), "Deploy to prod?", ["yes, deploy", "no, hold"], "cl-e2e", session_key)
    assert result.success and result.message_id == "100"
    sent = api.of("sendMessage")[0]
    keyboard = sent["reply_markup"].inline_keyboard
    assert [b.callback_data for row in keyboard for b in row] == ["cl:cl-e2e:0", "cl:cl-e2e:1", "cl:cl-e2e:other"]
    assert "1. yes, deploy" in sent["text"] and "2. no, hold" in sent["text"]

    # The agent thread is blocked in wait_for_response while the user taps.
    waiter = asyncio.get_running_loop().run_in_executor(None, cg.wait_for_response, "cl-e2e", 5.0)
    await asyncio.sleep(0.05)

    # A stranger's tap is refused and leaves the prompt armed.
    await adapter._handle_callback_query(
        _callback_update(api, "cl:cl-e2e:0", message=_bot_message(100, sent["text"]), user_id=STRANGER), None)
    assert "not authorized" in api.of("answerCallbackQuery")[-1]["text"]
    assert not waiter.done()

    # The owner's tap: answerCallbackQuery + keyboard removed + the choice text reaches the waiting tool.
    await adapter._handle_callback_query(
        _callback_update(api, "cl:cl-e2e:0", message=_bot_message(100, sent["text"])), None)
    assert await asyncio.wait_for(waiter, 5.0) == "yes, deploy"
    assert api.of("answerCallbackQuery")[-1]["text"] == "✓ yes, deploy"
    edit = api.of("editMessageText")[-1]
    assert edit["reply_markup"] is None and "yes, deploy" in edit["text"]
    assert rig.seen == []  # resolved the blocked tool call; no new agent turn


@pytest.mark.asyncio
async def test_late_clarify_tap_becomes_a_user_turn(rig):
    adapter, api = rig.adapter, rig.api
    prompt = _bot_message(100, "❓ Deploy to prod?\n\n1. yes, deploy\n2. no, hold")
    await adapter._handle_callback_query(_callback_update(api, "cl:cl-expired:1", message=prompt), None)
    await _drain(adapter)
    assert api.of("answerCallbackQuery")[-1]["text"] == "✓ no, hold"
    assert "expired" in api.of("editMessageText")[-1]["text"]
    assert len(rig.seen) == 1
    event = rig.seen[0]
    assert event.text == "no, hold" and event.reply_to_message_id == "100" and event.reply_to_is_own_message
    assert "Deploy to prod?" in event.reply_to_text
    assert any("On it: no, hold" in kw["text"] for kw in api.of("sendMessage"))


# ---------------------------------------------------------------------------
# 2. Reaction map round-trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_thumbs_up_on_bot_message_round_trips_as_yes(rig):
    adapter, api = rig.adapter, rig.api
    proposal = "Proposal: rotate the shim bearer tonight at 02:00. Go?"
    result = await adapter.send(str(CHAT), proposal)
    assert result.success and result.message_id == "100"

    await adapter._on_platform_update(_reaction_update(["👍"], message_id=100), None)
    await _drain(adapter)

    assert len(rig.seen) == 1
    event = rig.seen[0]
    assert event.text == "[reaction 👍] yes"
    assert event.reply_to_message_id == "100" and event.reply_to_is_own_message and event.reply_to_text == proposal
    assert event.source.user_id == str(OWNER) and event.source.chat_id == str(CHAT) and event.source.chat_type == "dm"
    # The agent's answer went back out through the API…
    assert any("On it:" in kw["text"] for kw in api.of("sendMessage")[1:])
    # …and the lifecycle 👀/👍 reactions did not land on the bot's own message.
    assert api.of("setMessageReaction") == []


@pytest.mark.asyncio
async def test_unmapped_emoji_and_stranger_start_nothing(rig):
    adapter, api = rig.adapter, rig.api
    await adapter.send(str(CHAT), "Anything to add?")
    await adapter._on_platform_update(_reaction_update(["🔥"], message_id=100), None)
    await adapter._on_platform_update(_reaction_update(["👍"], message_id=100, user_id=STRANGER), None)
    await adapter._on_platform_update(_reaction_update([], message_id=100, old=["👍"]), None)  # removal
    await _drain(adapter)
    assert rig.seen == []
    assert len(api.of("sendMessage")) == 1


@pytest.mark.asyncio
async def test_lifecycle_reactions_still_mark_real_user_messages(rig):
    """Sanity for the opposite direction: a normal inbound message gets 👀 then 👍."""
    from gateway.platforms.event import MessageEvent, MessageType
    adapter, api = rig.adapter, rig.api
    source = adapter.build_source(chat_id=str(CHAT), chat_type="dm", user_id=str(OWNER), user_name="John", message_id="55")
    await adapter.handle_message(MessageEvent(text="status?", message_type=MessageType.TEXT, source=source, message_id="55"))
    await _drain(adapter)
    reactions = [(kw["message_id"], kw["reaction"]) for kw in api.of("setMessageReaction")]
    assert reactions == [(55, "\U0001f440"), (55, "\U0001f44d")]
