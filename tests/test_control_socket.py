import asyncio
import json
import os
import socket
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from toad.app import ToadApp
from toad.cli import main
from toad.control_socket import (
    MAX_REQUEST_SIZE,
    ControlError,
    ControlSocketPathError,
    ControlSocketServer,
    ExternalPromptController,
    QueueResult,
    parse_request,
)
from toad.session_tracker import SessionDetails


class ProtocolTests(unittest.TestCase):
    def test_parse_valid_request_and_ignore_unknown_fields(self) -> None:
        request = parse_request(
            b'{"version":1,"id":"request-1","action":"prompt","future":true}'
        )

        self.assertEqual(request.id, "request-1")
        self.assertEqual(request.action, "prompt")
        self.assertTrue(request.body["future"])

    def test_parse_errors_have_stable_codes(self) -> None:
        cases = [
            (b"not json", "invalid_request"),
            (b"[]", "invalid_request"),
            (b'{"version":2,"id":"1","action":"ping"}', "unsupported_version"),
            (b'{"version":1,"id":"1","action":"launch"}', "unknown_action"),
            (b"x" * (MAX_REQUEST_SIZE + 1), "request_too_large"),
        ]

        for payload, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(ControlError) as caught:
                    parse_request(payload)
                self.assertEqual(caught.exception.code, code)


class PromptControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_priority_fifo_and_coalescing(self) -> None:
        submitted: list[str] = []

        async def submit(text: str) -> None:
            submitted.append(text)

        controller = ExternalPromptController(submit)
        accepted = await controller.enqueue(
            "active", coalesce_key="inbox", can_start=True
        )
        first = await controller.enqueue("normal-1", can_start=False)
        active_key_reused = await controller.enqueue(
            "old inbox", coalesce_key="inbox", can_start=False
        )
        await controller.enqueue("urgent", priority="urgent", can_start=False)
        coalesced = await controller.enqueue(
            "new inbox", coalesce_key="inbox", can_start=False
        )

        self.assertEqual(accepted.state, "accepted")
        self.assertEqual(first.state, "queued")
        self.assertEqual(active_key_reused.state, "queued")
        self.assertEqual(coalesced.state, "coalesced")
        self.assertEqual(coalesced.queue_depth, 3)
        self.assertEqual(submitted, ["active"])

        await controller.turn_finished(can_start=True)
        await controller.turn_finished(can_start=True)
        await controller.turn_finished(can_start=True)

        self.assertEqual(submitted, ["active", "urgent", "normal-1", "new inbox"])
        self.assertEqual(controller.queue_depth, 0)

    async def test_queue_is_bounded_but_coalescing_still_works(self) -> None:
        async def submit(_text: str) -> None:
            pass

        controller = ExternalPromptController(submit, queue_limit=1)
        await controller.enqueue("active", can_start=True)
        await controller.enqueue("old", coalesce_key="same", can_start=False)

        result = await controller.enqueue(
            "new", coalesce_key="same", can_start=False
        )
        self.assertEqual(result.state, "coalesced")
        with self.assertRaises(ControlError) as caught:
            await controller.enqueue("overflow", can_start=False)
        self.assertEqual(caught.exception.code, "queue_full")

    async def test_submission_does_not_touch_composer_state(self) -> None:
        composer = {"text": "partially written human draft", "focused": True}
        submitted: list[str] = []

        async def submit(text: str) -> None:
            submitted.append(text)

        controller = ExternalPromptController(submit)
        await controller.enqueue("external prompt", can_start=True)

        self.assertEqual(submitted, ["external prompt"])
        self.assertEqual(
            composer, {"text": "partially written human draft", "focused": True}
        )


class DispatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_dispatch_and_session_target_errors(self) -> None:
        class Conversation:
            control_session_id = "acp-session"
            control_state = "idle"
            external_queue_depth = 0
            agent_title = "Test Agent"
            current_mode = None

            def __init__(self) -> None:
                self.received: tuple[str, str, str | None] | None = None

            async def enqueue_external_prompt(
                self, text: str, *, priority: str, coalesce_key: str | None
            ) -> QueueResult:
                self.received = (text, priority, coalesce_key)
                return QueueResult("accepted", 0)

            async def cancel_control_turn(self) -> tuple[bool, bool]:
                return True, True

        app = ToadApp()
        app.session_tracker.sessions["session-1"] = SessionDetails(1, "session-1")
        conversation = Conversation()
        screen = SimpleNamespace(id="session-1")

        request = parse_request(
            b'{"version":1,"id":"1","action":"prompt","text":" wake ",'
            b'"priority":"urgent","coalesceKey":"inbox"}'
        )
        with patch.object(
            app, "_active_control_session", return_value=(screen, conversation)
        ):
            status = await app._handle_control_request(
                parse_request(b'{"version":1,"id":"0","action":"status"}')
            )
            self.assertEqual(status["sessionId"], "acp-session")
            self.assertEqual(status["state"], "idle")

            response = await app._handle_control_request(request)
            self.assertEqual(response["state"], "accepted")
            self.assertEqual(
                conversation.received, ("wake", "urgent", "inbox")
            )

            cancel = await app._handle_control_request(
                parse_request(b'{"version":1,"id":"c","action":"cancel"}')
            )
            self.assertTrue(cancel["active"])
            self.assertTrue(cancel["submitted"])

            unknown = parse_request(
                b'{"version":1,"id":"2","action":"prompt","text":"wake",'
                b'"sessionId":"unknown"}'
            )
            with self.assertRaises(ControlError) as caught:
                await app._handle_control_request(unknown)
            self.assertEqual(caught.exception.code, "unknown_session")

            app.session_tracker.sessions["session-2"] = SessionDetails(
                2, "session-2"
            )
            ambiguous = parse_request(
                b'{"version":1,"id":"3","action":"prompt","text":"wake"}'
            )
            with self.assertRaises(ControlError) as caught:
                await app._handle_control_request(ambiguous)
            self.assertEqual(caught.exception.code, "ambiguous_session")


class SocketTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    async def asyncTearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    async def handler(request) -> dict[str, object]:
        return {"action": request.action, "state": "idle", "queueDepth": 0}

    async def test_listener_roundtrip_and_cleanup(self) -> None:
        path = self.root / "missing" / "control.sock"
        server = ControlSocketServer(path, self.handler)
        await server.start()
        try:
            self.assertTrue(stat.S_ISSOCK(path.stat().st_mode))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)

            reader, writer = await asyncio.open_unix_connection(path)
            writer.write(
                b'{"version":1,"id":"roundtrip","action":"ping"}\n'
                b'{"version":1,"id":"ignored","action":"ping"}\n'
            )
            await writer.drain()
            response = json.loads(await reader.readline())
            self.assertEqual(
                response,
                {
                    "version": 1,
                    "id": "roundtrip",
                    "ok": True,
                    "action": "ping",
                    "state": "idle",
                    "queueDepth": 0,
                },
            )
            self.assertEqual(await reader.read(), b"")
            writer.close()
            await writer.wait_closed()

            reader, writer = await asyncio.open_unix_connection(path)
            writer.write(b"not json\n")
            await writer.drain()
            error = json.loads(await reader.readline())
            self.assertEqual(error["error"]["code"], "invalid_request")
            writer.close()
            await writer.wait_closed()

            reader, writer = await asyncio.open_unix_connection(path)
            writer.write(b"x" * (MAX_REQUEST_SIZE + 1) + b"\n")
            await writer.drain()
            error = json.loads(await reader.readline())
            self.assertEqual(error["error"]["code"], "request_too_large")
            writer.close()
            await writer.wait_closed()

            reader, writer = await asyncio.open_unix_connection(path)
            writer.write(b'{"version":1,"id":"still-alive","action":"status"}\n')
            await writer.drain()
            response = json.loads(await reader.readline())
            self.assertTrue(response["ok"])
            self.assertEqual(response["id"], "still-alive")
            writer.close()
            await writer.wait_closed()
        finally:
            await server.stop()

        self.assertFalse(path.exists())

    async def test_non_socket_collision_is_preserved(self) -> None:
        path = self.root / "control.sock"
        path.write_text("keep me", encoding="utf-8")
        server = ControlSocketServer(path, self.handler)

        with self.assertRaises(ControlSocketPathError):
            await server.start()

        self.assertEqual(path.read_text(encoding="utf-8"), "keep me")

    async def test_stale_socket_is_replaced(self) -> None:
        path = self.root / "control.sock"
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(os.fspath(path))
        stale.close()

        server = ControlSocketServer(path, self.handler)
        await server.start()
        self.assertTrue(stat.S_ISSOCK(path.stat().st_mode))
        await server.stop()
        self.assertFalse(path.exists())

    async def test_cleanup_preserves_replacement_path(self) -> None:
        path = self.root / "control.sock"
        server = ControlSocketServer(path, self.handler)
        await server.start()
        path.unlink()
        path.write_text("replacement", encoding="utf-8")

        await server.stop()

        self.assertEqual(path.read_text(encoding="utf-8"), "replacement")

    async def test_live_socket_is_not_replaced(self) -> None:
        path = self.root / "control.sock"
        live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        live.bind(os.fspath(path))
        live.listen()
        try:
            server = ControlSocketServer(path, self.handler)
            with self.assertRaises(ControlSocketPathError):
                await server.start()
            self.assertTrue(stat.S_ISSOCK(path.stat().st_mode))
        finally:
            live.close()
            path.unlink()

    async def test_listener_times_out_incomplete_request(self) -> None:
        path = self.root / "control.sock"
        server = ControlSocketServer(path, self.handler)
        await server.start()
        try:
            reader, writer = await asyncio.open_unix_connection(path)
            with patch("toad.control_socket.READ_TIMEOUT", 0.01):
                error = json.loads(await reader.readline())
            self.assertEqual(error["error"]["code"], "invalid_request")
            writer.close()
            await writer.wait_closed()
        finally:
            await server.stop()


class CliTests(unittest.TestCase):
    def test_control_socket_option_is_available_on_both_commands(self) -> None:
        runner = CliRunner()

        run_help = runner.invoke(main, ["run", "--help"])
        acp_help = runner.invoke(main, ["acp", "--help"])

        self.assertEqual(run_help.exit_code, 0)
        self.assertEqual(acp_help.exit_code, 0)
        self.assertIn("--control-socket PATH", run_help.output)
        self.assertIn("--control-socket PATH", acp_help.output)

    def test_control_socket_rejects_web_serve_mode(self) -> None:
        runner = CliRunner()

        run_result = runner.invoke(
            main, ["run", "--serve", "--control-socket", "/tmp/toad.sock"]
        )
        acp_result = runner.invoke(
            main,
            [
                "acp",
                "agent acp",
                "--serve",
                "--control-socket",
                "/tmp/toad.sock",
            ],
        )

        self.assertEqual(run_result.exit_code, 2)
        self.assertEqual(acp_result.exit_code, 2)
        self.assertIn("cannot be combined", run_result.output)
        self.assertIn("cannot be combined", acp_result.output)


if __name__ == "__main__":
    unittest.main()
