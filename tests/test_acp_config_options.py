from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
import unittest
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from toad.acp import api, messages, protocol
from toad.acp.agent import Agent, Mode
from toad.agent_schema import Agent as AgentData
from toad.app import ToadApp
from toad.control_socket import ControlError, parse_request
from toad.session_tracker import SessionDetails
from toad.widgets.conversation import Conversation


def select_option(
    config_id: str = "model",
    current: str = "small",
    *,
    category: str = "model",
) -> protocol.SessionConfigSelect:
    return {
        "id": config_id,
        "name": config_id.title(),
        "category": category,
        "type": "select",
        "currentValue": current,
        "options": [
            {"value": "small", "name": "Small"},
            {"value": "large", "name": "Large"},
        ],
        "futureField": {"preserved": True},
    }


def boolean_option() -> protocol.SessionConfigBoolean:
    return {
        "id": "brave",
        "name": "Brave",
        "type": "boolean",
        "currentValue": False,
    }


class FakeCall:
    def __init__(self, result: Any) -> None:
        self.result = result

    async def wait(self) -> Any:
        return self.result


def make_agent(session_id: str | None = None) -> Agent:
    data = cast(
        AgentData,
        {
            "name": "Test Agent",
            "identity": "test.example",
            "run_command": {"*": "test-agent"},
        },
    )
    agent = Agent(Path("."), data, session_id)
    agent.request = MagicMock(return_value=nullcontext())
    return agent


class AgentConfigOptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_advertises_boolean_config_options(self) -> None:
        agent = make_agent()
        with patch(
            "toad.acp.agent.api.initialize",
            return_value=FakeCall({"protocolVersion": 1}),
        ) as initialize:
            await agent.acp_initialize()

        capabilities = initialize.call_args.args[1]
        self.assertEqual(
            capabilities["session"]["configOptions"]["boolean"],
            {},
        )

    async def test_new_and_load_replace_and_publish_config_options(self) -> None:
        agent = make_agent()
        posted: list[object] = []
        agent._message_target = SimpleNamespace(
            post_message=lambda message: posted.append(message) or True
        )

        initial = [select_option()]
        with patch(
            "toad.acp.agent.api.session_new",
            return_value=FakeCall(
                {"sessionId": "session-1", "configOptions": initial}
            ),
        ):
            await agent.acp_new_session()
        self.assertEqual(agent.config_options, initial)
        self.assertIsInstance(posted[-1], messages.ConfigOptionsUpdate)

        loaded = [boolean_option()]
        with patch(
            "toad.acp.agent.api.session_load",
            return_value=FakeCall({"configOptions": loaded}),
        ):
            await agent.acp_load_session()
        self.assertEqual(agent.config_options, loaded)
        self.assertEqual(posted[-1].config_options, loaded)

    async def test_notification_and_set_response_are_full_replacements(self) -> None:
        agent = make_agent("session-1")
        posted: list[object] = []
        agent._message_target = SimpleNamespace(
            post_message=lambda message: posted.append(message) or True
        )
        agent.config_options = [select_option(), boolean_option()]

        notification = [select_option(current="large")]
        agent.rpc_session_update(
            "session-1",
            {
                "sessionUpdate": "config_option_update",
                "configOptions": notification,
            },
        )
        self.assertEqual(agent.config_options, notification)

        dependent = [
            select_option(current="large"),
            {
                "id": "effort",
                "name": "Effort",
                "category": "thought_level",
                "type": "select",
                "currentValue": "large",
                "options": [{"value": "large", "name": "Large only"}],
            },
        ]
        with patch(
            "toad.acp.agent.api.session_set_config_option",
            return_value=FakeCall({"configOptions": dependent}),
        ) as set_config:
            confirmed = await agent.set_config_option("model", "large")

        set_config.assert_called_once_with("session-1", "model", "large")
        self.assertEqual(confirmed, dependent)
        self.assertEqual(agent.config_options, dependent)
        self.assertEqual(posted[-1].config_options, dependent)

    async def test_api_uses_exact_stable_method_and_params(self) -> None:
        with api.API.request() as request:
            api.session_set_config_option("session-1", "model", "large")
        self.assertEqual(
            request.body,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/set_config_option",
                "params": {
                    "sessionId": "session-1",
                    "configId": "model",
                    "value": "large",
                },
            },
        )

        with api.API.request() as request:
            api.session_set_config_option("session-1", "brave", True, "boolean")
        self.assertEqual(
            request.body["params"],
            {
                "sessionId": "session-1",
                "configId": "brave",
                "value": True,
                "type": "boolean",
            },
        )


class ConversationConfigOptionTests(unittest.IsolatedAsyncioTestCase):
    def make_conversation(self) -> Conversation:
        return Conversation(Path("."))

    async def test_generic_picker_renders_select_and_boolean_values(self) -> None:
        app = ToadApp()
        async with app.run_test() as pilot:
            conversation = app.screen.conversation
            conversation.config_options = [select_option(), boolean_option()]

            self.assertTrue(conversation.open_config_picker("model"))
            await pilot.pause()
            picker = conversation.prompt.config_switcher
            self.assertTrue(picker.has_focus)
            self.assertEqual(set(picker._values.values()), {"small", "large"})

            picker.open([boolean_option()])
            self.assertEqual(set(picker._values.values()), {True, False})

    async def test_select_boolean_validation_and_idle_rejection(self) -> None:
        conversation = self.make_conversation()
        dependent = [select_option(current="large")]
        agent = SimpleNamespace(
            session_id="session-1",
            set_config_option=AsyncMock(return_value=dependent),
        )
        conversation.set_reactive(Conversation.agent, agent)
        conversation.set_reactive(
            Conversation.config_options, [select_option(), boolean_option()]
        )

        with (
            patch.object(
                Conversation,
                "control_state",
                new_callable=PropertyMock,
                return_value="idle",
            ),
            patch.object(conversation, "update_slash_commands"),
        ):
            conversation.set_reactive(Conversation.config_options, [])
            with self.assertRaises(ControlError) as caught:
                await conversation.set_config_option("model", "large")
            self.assertEqual(caught.exception.code, "config_options_unsupported")
            conversation.set_reactive(
                Conversation.config_options, [select_option(), boolean_option()]
            )

            self.assertEqual(
                await conversation.set_config_option("model", "large"), dependent
            )
            conversation.set_reactive(
                Conversation.config_options, [select_option(), boolean_option()]
            )
            await conversation.set_config_option("brave", True)
            conversation.set_reactive(
                Conversation.config_options, [select_option(), boolean_option()]
            )

            for config_id, value, code in (
                ("missing", "large", "unknown_config_option"),
                ("model", "invented", "invalid_config_value"),
                ("model", True, "invalid_config_value"),
                ("brave", "true", "invalid_config_value"),
            ):
                with self.subTest(config_id=config_id, value=value):
                    with self.assertRaises(ControlError) as caught:
                        await conversation.set_config_option(config_id, value)
                    self.assertEqual(caught.exception.code, code)

        with patch.object(
            Conversation, "control_state", new_callable=PropertyMock, return_value="busy"
        ):
            with self.assertRaises(ControlError) as caught:
                await conversation.set_config_option("model", "large")
            self.assertEqual(caught.exception.code, "not_ready")

        conversation.quiesce_external_prompts()
        with patch.object(
            Conversation, "control_state", new_callable=PropertyMock, return_value="idle"
        ):
            with self.assertRaises(ControlError) as caught:
                await conversation.set_config_option("model", "large")
            self.assertEqual(caught.exception.code, "not_ready")

    async def test_legacy_mode_fallback_and_config_mode_precedence(self) -> None:
        conversation = self.make_conversation()
        agent = SimpleNamespace(
            session_id="session-1",
            set_mode=AsyncMock(return_value=None),
        )
        conversation.set_reactive(Conversation.agent, agent)
        conversation.set_reactive(
            Conversation.modes, {"code": Mode("code", "Code", None)}
        )

        with patch.object(conversation, "flash"):
            await conversation.set_mode("code")
        agent.set_mode.assert_awaited_once_with("code")

        conversation.set_reactive(
            Conversation.config_options,
            [select_option("session-mode", category="mode")],
        )
        with patch.object(
            conversation,
            "set_config_option",
            AsyncMock(return_value=conversation.config_options),
        ) as set_config:
            await conversation.set_mode("large")
        set_config.assert_awaited_once_with("session-mode", "large")

    async def test_local_config_command_never_reaches_agent_prompt(self) -> None:
        conversation = self.make_conversation()
        conversation.set_reactive(Conversation.config_options, [select_option()])
        conversation.set_reactive(
            Conversation.agent,
            SimpleNamespace(
                session_id="session-1",
                send_prompt=AsyncMock(),
            ),
        )
        with patch.object(
            conversation, "open_config_picker", return_value=True
        ) as open_picker:
            handled = await conversation.slash_command("/model")

        self.assertTrue(handled)
        open_picker.assert_called_once_with("model")
        conversation.agent.send_prompt.assert_not_awaited()


class ControlConfigOptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_control_config_roundtrip_and_status_summary(self) -> None:
        options = [select_option(), boolean_option()]

        class ControlConversation:
            control_session_id = "session-1"
            control_state = "idle"
            external_queue_depth = 0
            external_prompts_accepting = True
            control_resume_supported = True
            agent_title = "Test Agent"
            current_mode = None
            config_options = options

            async def set_config_option(self, config_id, value):
                self.received = (config_id, value)
                self.config_options = [
                    select_option(current="large"),
                    select_option(
                        "effort", current="large", category="thought_level"
                    ),
                    select_option("mode", current="large", category="mode"),
                ]
                return self.config_options

        app = ToadApp()
        app.session_tracker.sessions["screen-1"] = SessionDetails(1, "screen-1")
        conversation = ControlConversation()
        screen = SimpleNamespace(id="screen-1")

        with patch.object(
            app, "_active_control_session", return_value=(screen, conversation)
        ):
            listed = await app._handle_control_request(
                parse_request(
                    b'{"version":1,"id":"1","action":"configOptions",'
                    b'"sessionId":"session-1"}'
                )
            )
            self.assertEqual(listed["configOptions"], options)

            changed = await app._handle_control_request(
                parse_request(
                    b'{"version":1,"id":"2","action":"setConfig",'
                    b'"sessionId":"session-1","configId":"model","value":"large"}'
                )
            )
            self.assertEqual(conversation.received, ("model", "large"))
            self.assertEqual(changed["configOptions"], conversation.config_options)

            status = await app._handle_control_request(
                parse_request(b'{"version":1,"id":"3","action":"status"}')
            )
            self.assertEqual(status["model"], "large")
            self.assertEqual(status["effort"], "large")
            self.assertEqual(status["mode"], "large")
            self.assertNotIn("configOptions", status)


if __name__ == "__main__":
    unittest.main()
