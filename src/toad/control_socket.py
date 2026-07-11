from __future__ import annotations

import asyncio
import errno
import json
import os
import socket
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, cast


PROTOCOL_VERSION = 1
MAX_REQUEST_SIZE = 64 * 1024
MAX_QUEUE_DEPTH = 100
READ_TIMEOUT = 5.0

type ControlAction = Literal["ping", "status", "prompt", "cancel", "quiesce"]
type PromptPriority = Literal["normal", "urgent"]
type QueueState = Literal["accepted", "queued", "coalesced"]


class ControlError(Exception):
    """An error that may be returned over the control protocol."""

    def __init__(
        self, code: str, message: str, *, request_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id


class ControlSocketPathError(RuntimeError):
    """The control socket path can not be used safely."""


@dataclass(frozen=True, slots=True)
class ControlRequest:
    id: str
    action: ControlAction
    body: dict[str, Any]


def parse_request(data: bytes) -> ControlRequest:
    """Parse and validate one control protocol request."""

    if len(data) > MAX_REQUEST_SIZE:
        raise ControlError("request_too_large", "request exceeds 64 KiB")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlError("invalid_request", "request must be UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ControlError("invalid_request", "request must be a JSON object")

    request_id = value.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise ControlError("invalid_request", "request id is required")

    if "version" not in value:
        raise ControlError(
            "invalid_request", "protocol version is required", request_id=request_id
        )
    if type(value["version"]) is not int or value["version"] != PROTOCOL_VERSION:
        raise ControlError(
            "unsupported_version",
            f"unsupported protocol version: {value['version']!r}",
            request_id=request_id,
        )

    action = value.get("action")
    if not isinstance(action, str) or not action:
        raise ControlError(
            "invalid_request", "action is required", request_id=request_id
        )
    if action not in {"ping", "status", "prompt", "cancel", "quiesce"}:
        raise ControlError(
            "unknown_action", f"unknown action: {action}", request_id=request_id
        )
    return ControlRequest(request_id, cast(ControlAction, action), value)


@dataclass(frozen=True, slots=True)
class ExternalPrompt:
    text: str
    priority: PromptPriority = "normal"
    coalesce_key: str | None = None


@dataclass(frozen=True, slots=True)
class QueueResult:
    state: QueueState
    queue_depth: int


class ExternalPromptController:
    """Own an external prompt queue without interacting with the composer."""

    def __init__(
        self,
        submit_prompt: Callable[[str], Awaitable[None]],
        *,
        queue_limit: int = MAX_QUEUE_DEPTH,
    ) -> None:
        self._submit_prompt = submit_prompt
        self._queue_limit = queue_limit
        self._queue: list[ExternalPrompt] = []
        self._active: ExternalPrompt | None = None
        self._accepting = True

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def has_active(self) -> bool:
        return self._active is not None

    @property
    def accepting(self) -> bool:
        return self._accepting

    def quiesce(self) -> None:
        """Atomically reject new external prompts for process replacement."""
        self._accepting = False

    async def enqueue(
        self,
        text: str,
        *,
        priority: PromptPriority = "normal",
        coalesce_key: str | None = None,
        can_start: bool,
    ) -> QueueResult:
        if not self._accepting:
            raise ControlError("quiesced", "external prompt delivery is quiesced")

        prompt = ExternalPrompt(text, priority, coalesce_key)

        if coalesce_key is not None:
            for index, queued in enumerate(self._queue):
                if queued.coalesce_key == coalesce_key:
                    del self._queue[index]
                    self._queue.append(prompt)
                    return QueueResult("coalesced", self.queue_depth)

        if can_start and self._active is None and not self._queue:
            await self._start(prompt)
            return QueueResult("accepted", self.queue_depth)

        if self.queue_depth >= self._queue_limit:
            raise ControlError("queue_full", "external prompt queue is full")
        self._queue.append(prompt)
        return QueueResult("queued", self.queue_depth)

    async def turn_finished(self, *, can_start: bool) -> bool:
        """Finish the current turn and start at most one queued prompt."""

        self._active = None
        if not can_start or not self._queue:
            return False
        urgent_index = next(
            (
                index
                for index, prompt in enumerate(self._queue)
                if prompt.priority == "urgent"
            ),
            0,
        )
        prompt = self._queue.pop(urgent_index)
        await self._start(prompt)
        return True

    async def _start(self, prompt: ExternalPrompt) -> None:
        self._active = prompt
        try:
            await self._submit_prompt(prompt.text)
        except BaseException:
            self._active = None
            raise


type RequestHandler = Callable[[ControlRequest], Awaitable[dict[str, object]]]


class ControlSocketServer:
    """A bounded, one-request-per-connection Unix socket server."""

    def __init__(self, path: str | Path, handler: RequestHandler) -> None:
        self.path = Path(path).expanduser().resolve()
        self._handler = handler
        self._server: asyncio.Server | None = None
        self._owned_identity: tuple[int, int] | None = None
        self._client_tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        if self._server is not None:
            return
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not parent_existed:
            self.path.parent.chmod(0o700)
        self._remove_stale_socket()

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            previous_umask = os.umask(0o177)
            try:
                listener.bind(os.fspath(self.path))
            finally:
                os.umask(previous_umask)
            listener.listen()
            socket_stat = self.path.lstat()
            self._owned_identity = (socket_stat.st_dev, socket_stat.st_ino)
            self.path.chmod(0o600)
            server = await asyncio.start_unix_server(
                self._client_connected,
                sock=listener,
                cleanup_socket=False,
            )
        except BaseException:
            listener.close()
            if "server" in locals():
                server.close()
                await server.wait_closed()
            self._remove_owned_socket()
            raise
        self._server = server

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        tasks = tuple(self._client_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._remove_owned_socket()

    def _remove_stale_socket(self) -> None:
        try:
            socket_stat = self.path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(socket_stat.st_mode):
            raise ControlSocketPathError(
                f"refusing to replace non-socket path: {self.path}"
            )
        if self._socket_is_live():
            raise ControlSocketPathError(
                f"a live listener already owns control socket: {self.path}"
            )

        current_stat = self.path.lstat()
        if (
            current_stat.st_dev,
            current_stat.st_ino,
            stat.S_IFMT(current_stat.st_mode),
        ) != (
            socket_stat.st_dev,
            socket_stat.st_ino,
            stat.S_IFSOCK,
        ):
            raise ControlSocketPathError(
                f"control socket path changed during startup: {self.path}"
            )
        self.path.unlink()

    def _socket_is_live(self) -> bool:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            try:
                result = probe.connect_ex(os.fspath(self.path))
            except TimeoutError:
                return True
        if result == 0:
            return True
        if result in {errno.ECONNREFUSED, errno.ENOENT}:
            return False
        raise ControlSocketPathError(
            f"unable to inspect control socket {self.path}: {os.strerror(result)}"
        )

    def _remove_owned_socket(self) -> None:
        owned_identity = self._owned_identity
        self._owned_identity = None
        if owned_identity is None:
            return
        try:
            socket_stat = self.path.lstat()
        except FileNotFoundError:
            return
        if (
            socket_stat.st_dev,
            socket_stat.st_ino,
        ) == owned_identity and stat.S_ISSOCK(socket_stat.st_mode):
            self.path.unlink()

    def _client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.create_task(self._handle_client(reader, writer))
        self._client_tasks.add(task)
        task.add_done_callback(self._client_tasks.discard)

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        request: ControlRequest | None = None
        try:
            try:
                async with asyncio.timeout(READ_TIMEOUT):
                    data = await self._read_line(reader)
            except TimeoutError as error:
                raise ControlError(
                    "invalid_request", "request line timed out"
                ) from error
            request = parse_request(data)
            payload = await self._handler(request)
            response: dict[str, object] = {
                "version": PROTOCOL_VERSION,
                "id": request.id,
                "ok": True,
                **payload,
            }
        except ControlError as error:
            response = {
                "version": PROTOCOL_VERSION,
                "id": error.request_id or (request.id if request else None),
                "ok": False,
                "error": {"code": error.code, "message": error.message},
            }
        except Exception:
            response = {
                "version": PROTOCOL_VERSION,
                "id": request.id if request else None,
                "ok": False,
                "error": {
                    "code": "internal_error",
                    "message": "internal control socket error",
                },
            }

        try:
            writer.write(
                json.dumps(response, separators=(",", ":"), ensure_ascii=False).encode(
                    "utf-8"
                )
                + b"\n"
            )
            await writer.drain()
        except (ConnectionError, BrokenPipeError):
            pass
        finally:
            writer.close()
            with suppress(ConnectionError, BrokenPipeError):
                await writer.wait_closed()

    @staticmethod
    async def _read_line(reader: asyncio.StreamReader) -> bytes:
        data = bytearray()
        while True:
            chunk = await reader.read(min(4096, MAX_REQUEST_SIZE + 1 - len(data)))
            if not chunk:
                if not data:
                    raise ControlError("invalid_request", "request line is required")
                raise ControlError(
                    "invalid_request", "request must end with a newline"
                )
            newline = chunk.find(b"\n")
            if newline >= 0:
                data.extend(chunk[:newline])
                if len(data) > MAX_REQUEST_SIZE:
                    raise ControlError("request_too_large", "request exceeds 64 KiB")
                return bytes(data)
            data.extend(chunk)
            if len(data) > MAX_REQUEST_SIZE:
                raise ControlError("request_too_large", "request exceeds 64 KiB")
