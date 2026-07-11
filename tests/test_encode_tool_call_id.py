from __future__ import annotations

import unittest

from click.testing import CliRunner
from textual.dom import check_identifiers
from toad.acp.encode_tool_call_id import encode_tool_call_id
from toad.cli import main


class EncodeToolCallIdTests(unittest.TestCase):
    def test_newline_separated_cursor_id_is_valid_textual_id(self) -> None:
        raw_id = "call_example\nfc_example"

        encoded = encode_tool_call_id(raw_id)

        self.assertEqual(encoded, "tool-call-63616C6C5F6578616D706C650A66635F6578616D706C65")
        check_identifiers("id", encoded)

    def test_acp_help_exposes_compact_ui_mode(self) -> None:
        result = CliRunner().invoke(main, ["acp", "--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--compact-ui", result.output)
        self.assertIn("--session-id", result.output)


if __name__ == "__main__":
    unittest.main()
