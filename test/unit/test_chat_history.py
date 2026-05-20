"""Unit tests for chat_history — pop_last_assistant_turn (reroll logic) and
dialog content parsing."""

from unittest.mock import MagicMock
import sys

sys.modules.setdefault("PySide6", MagicMock())
sys.modules.setdefault("PySide6.QtWidgets", MagicMock())
sys.modules.setdefault("PySide6.QtCore", MagicMock())

from llm.history_manager import (
    _repair_json_string,
    parse_assistant_dialog_content,
)
from core.sprite.chat_history import pop_last_assistant_turn


def _uh(msg: str) -> str:
    return f"<b>你</b>：{msg}"


def _ah(name: str, msg: str) -> str:
    return f"<b>{name}</b>：{msg}"


def _u(msg: str) -> dict:
    return {"role": "user", "content": msg}


def _a(msg: str) -> dict:
    return {"role": "assistant", "content": msg}


def _t(call_id: str, name: str, result: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": result, "name": name}


class TestPopLastAssistantTurn:
    def test_single_turn_pops_both(self):
        hist = [_uh("hello"), _ah("Alice", "hi")]
        msgs = [_u("hello"), _a("hi")]
        last = pop_last_assistant_turn(hist, msgs)
        assert last == _uh("hello")
        assert hist == []
        assert msgs == []

    def test_second_turn_pops_only_last(self):
        hist = [_uh("hi"), _ah("Bob", "hey"), _uh("how are you"), _ah("Bob", "good")]
        msgs = [_u("hi"), _a("hey"), _u("how are you"), _a("good")]
        last = pop_last_assistant_turn(hist, msgs)
        assert last == _uh("how are you")
        assert hist == [_uh("hi"), _ah("Bob", "hey")]
        assert msgs == [_u("hi"), _a("hey")]

    def test_with_tool_calls(self):
        hist = [_uh("search"), _ah("Bot", "let me check"), _ah("Bot", "found")]
        msgs = [_u("search"), _a("let me check"), _t("c1", "search", "ok"), _a("found")]
        last = pop_last_assistant_turn(hist, msgs)
        assert last == _uh("search")
        assert hist == []
        assert msgs == []

    def test_no_assistant_yet(self):
        hist = [_uh("waiting")]
        msgs = [_u("waiting")]
        last = pop_last_assistant_turn(hist, msgs)
        assert last == _uh("waiting")
        assert hist == []
        assert msgs == []

    def test_empty(self):
        last = pop_last_assistant_turn([], [])
        assert last == ""

    def test_no_user_at_all(self):
        hist = [_ah("Bot", "hello")]
        msgs = [_a("hello")]
        last = pop_last_assistant_turn(hist, msgs)
        assert last == ""
        assert hist == []
        assert msgs == []


class TestParseAssistantDialogContent:

    def test_plain_json_object(self):
        dialog = parse_assistant_dialog_content(
            '{"dialog": [{"character_name": "Alice", "sprite": "1", '
            '"speech": "hello"}]}'
        )
        assert len(dialog) == 1
        assert dialog[0]["character_name"] == "Alice"

    def test_fenced_json(self):
        dialog = parse_assistant_dialog_content(
            '```json\n{"dialog": [{"character_name": "Bob", "sprite": "2", '
            '"speech": "hi"}]}\n```'
        )
        assert len(dialog) == 1
        assert dialog[0]["character_name"] == "Bob"

    def test_none_returns_empty(self):
        assert parse_assistant_dialog_content(None) == []

    def test_empty_string_returns_empty(self):
        assert parse_assistant_dialog_content("") == []

    def test_non_dict_returns_empty(self):
        assert parse_assistant_dialog_content('["just", "an", "array"]') == []

    def test_no_dialog_key_returns_empty(self):
        assert parse_assistant_dialog_content('{"other": 1}') == []

    def test_raw_newline_inside_speech_is_escaped(self):
        content = (
            '{"dialog": [{"character_name": "Alice", "sprite": "1", '
            '"speech": "line 1\nline 2"}]}'
        )
        dialog = parse_assistant_dialog_content(content)
        assert len(dialog) == 1
        assert dialog[0]["character_name"] == "Alice"
        assert "\n" in dialog[0]["speech"]

    def test_raw_tab_inside_speech_is_escaped(self):
        content = (
            '{"dialog": [{"character_name": "A", "sprite": "1", '
            '"speech": "col1\tcol2"}]}'
        )
        dialog = parse_assistant_dialog_content(content)
        assert len(dialog) == 1

    def test_missing_closing_quote_before_next_item(self):
        content = (
            '{\n  "dialog": [\n'
            '    {"character_name": "COT", "sprite": "-1", '
            '"speech": "<summary>long text</summary>\n'
            '    },\n'
            '    {"character_name": "Alice", "sprite": "1", '
            '"speech": "hello"}\n'
            '  ]\n}'
        )
        dialog = parse_assistant_dialog_content(content)
        assert len(dialog) == 2
        assert dialog[0]["character_name"] == "COT"
        assert dialog[1]["character_name"] == "Alice"

    def test_speech_missing_quote_and_closing_braces_is_unfixable(self):
        content = (
            '{"dialog": [{"character_name": "Alice", "sprite": "1", '
            '"speech": "hello'
        )
        dialog = parse_assistant_dialog_content(content)
        assert dialog == []

    def test_missing_quote_before_array_close(self):
        content = (
            '```json\n{\n  "dialog": [\n'
            '    {"character_name": "Narrator", "sprite": "-1", '
            '"speech": "the end'
            '\n  ]\n}\n```'
        )
        dialog = parse_assistant_dialog_content(content)
        assert dialog == []

    def test_newline_before_close_brace_is_structural(self):
        content = (
            '{"dialog": [{"character_name": "X", "sprite": "-1", '
            '"speech": "some text"\n    }\n  ]}'
        )
        dialog = parse_assistant_dialog_content(content)
        assert len(dialog) == 1

    def test_newline_before_open_brace_is_structural(self):
        content = (
            '{"dialog": [{"character_name": "X", "sprite": "-1", '
            '"speech": "some text"\n    }, {"character_name": "Y", '
            '"sprite": "2", "speech": "ok"}]}'
        )
        dialog = parse_assistant_dialog_content(content)
        assert len(dialog) == 2

    def test_cot_with_tags_and_newlines(self):
        content = (
            '{"dialog": ['
            '{"character_name": "COT", "sprite": "-1", '
            '"speech": "'
            '<summary>water island said something</summary>'
            '<motivation>fangshi: 1.heard her - paused</motivation>'
            '<plot>they continue walking</plot>'
            '"\n    },\n    {"character_name": "Alice", "sprite": "1", '
            '"speech": "ok"}\n  ]\n}'
        )
        dialog = parse_assistant_dialog_content(content)
        assert len(dialog) == 2
        assert dialog[0]["character_name"] == "COT"
        assert "summary" in dialog[0]["speech"]
        assert dialog[1]["character_name"] == "Alice"


class TestRepairJsonString:

    def test_idempotent_on_valid_json(self):
        valid = '{"key": "value", "num": 1}'
        assert _repair_json_string(valid) == valid

    def test_closes_unclosed_string_at_end(self):
        repaired = _repair_json_string('{"key": "value')
        assert repaired.endswith('"')

    def test_escapes_control_chars_inside_string(self):
        repaired = _repair_json_string('{"key": "val\x00ue"}')
        assert "\\u0000" in repaired

    def test_closes_before_structural_brace(self):
        repaired = _repair_json_string(
            '{"dialog": [{"speech": "text\n    }\n  ]\n}'
        )
        assert '"text"\n    }' in repaired

    def test_closes_before_structural_bracket(self):
        repaired = _repair_json_string(
            '{"dialog": [{"speech": "text\n  ]\n}'
        )
        assert '"text"\n  ]' in repaired

    def test_escaped_quote_not_confused(self):
        repaired = _repair_json_string(
            '{"speech": "he said \\"hello\\""}'
        )
        assert repaired.count('"') % 2 == 0
