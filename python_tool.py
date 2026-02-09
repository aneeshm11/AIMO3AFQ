import contextlib
import queue
import threading
import time
from pathlib import Path

from openai_harmony import Author, Message, Role, TextContent, ToolNamespaceConfig


def printify_tail(source: str) -> str:
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    lines = source.strip().splitlines()
    if not lines:
        return source
    tail = lines[-1].strip()
    if not tail or tail.startswith(("import ", "from ")) or "print(" in tail:
        return "\n".join(lines)
    tail = tail.split("#", 1)[0].rstrip()
    if not tail:
        return "\n".join(lines)
    lines[-1] = f"print({tail})"
    return "\n".join(lines)


class PortBatcher:
    _lock = threading.Lock()
    _cursor = 50000

    @classmethod
    def take(cls, n: int) -> list[int]:
        if n <= 0:
            raise ValueError("n must be positive")
        with cls._lock:
            start = cls._cursor
            cls._cursor += n
        return list(range(start, start + n))


class StatefulKernelSession:
    def __init__(self, connection_file: str | None = None, *, timeout: float = 120.0, kernel_name: str = "python3"):
        try:
            from jupyter_client import BlockingKernelClient, KernelManager
        except ImportError as exc:
            raise RuntimeError("jupyter_client package required") from exc

        self._timeout = float(timeout)
        self._owns_kernel = False
        self._km = None

        if connection_file:
            p = Path(connection_file).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"Connection file not found: {p}")
            client = BlockingKernelClient()
            client.load_connection_file(str(p))
            client.start_channels()
            client.wait_for_ready(timeout=self._timeout)
            self._client = client
            return

        ports = PortBatcher.take(5)
        km = KernelManager(kernel_name=kernel_name)
        km.shell_port, km.iopub_port, km.stdin_port, km.hb_port, km.control_port = ports
        km.start_kernel()
        client = km.blocking_client()
        client.start_channels()
        client.wait_for_ready(timeout=self._timeout)

        self._client = client
        self._km = km
        self._owns_kernel = True

    def run(self, code: str, *, timeout: float | None = None) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be a string")

        t = self._timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + t

        msg_id = self._client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)

        out: list[str] = []
        err: list[str] = []

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        def append_text(buf: list[str], text: str):
            if text:
                buf.append(text)

        while True:
            if remaining() <= 0:
                raise TimeoutError("Timed out waiting for kernel output.")
            try:
                msg = self._client.get_iopub_msg(timeout=min(0.25, remaining()))
            except queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {}) or {}

            if msg_type == "stream":
                name = content.get("name")
                text = content.get("text", "")
                append_text(out if name == "stdout" else err, text)
                continue

            if msg_type == "error":
                tb = content.get("traceback") or []
                if tb:
                    append_text(err, "\n".join(tb))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    append_text(err, (f"{ename}: {evalue}").strip())
                continue

            if msg_type in ("execute_result", "display_data"):
                data = content.get("data", {}) or {}
                text = data.get("text/plain")
                if isinstance(text, str) and text:
                    append_text(out, text if text.endswith("\n") else text + "\n")
                continue

            if msg_type == "status" and content.get("execution_state") == "idle":
                break

        while True:
            if remaining() <= 0:
                raise TimeoutError("Timed out waiting for execution reply.")
            try:
                reply = self._client.get_shell_msg(timeout=min(0.25, remaining()))
            except queue.Empty:
                continue

            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            rc = reply.get("content", {}) or {}
            if rc.get("status") == "error":
                tb = rc.get("traceback") or []
                if tb:
                    append_text(err, "\n".join(tb))
                else:
                    ename = rc.get("ename", "")
                    evalue = rc.get("evalue", "")
                    append_text(err, (f"{ename}: {evalue}").strip())
            break

        stdout = "".join(out).rstrip("\n")
        stderr = "".join(err).rstrip("\n")

        if stderr:
            merged = f"{stdout}\n{stderr}" if stdout else stderr
        else:
            merged = stdout

        if not merged.strip():
            merged = "[WARN] No output. Use print() to see results."

        return merged

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False

    def __del__(self):
        with contextlib.suppress(Exception):
            self.shutdown()


class JupyterPythonTool:
    def __init__(self, execution_backend: str | None = None, local_timeout: float = 80.0):
        self._local_timeout = float(local_timeout)
        self._exec_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._session: StatefulKernelSession | None = None

    @classmethod
    def tool_id(cls) -> str:
        return "python"

    @property
    def name(self) -> str:
        return self.tool_id()

    @property
    def instruction(self) -> str:
        return "Use this tool to execute Python code. The code runs in a stateful Jupyter notebook. Use print() to see output."

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(name=self.tool_id(), description=self.instruction, tools=[])

    def _session_or_create(self) -> StatefulKernelSession:
        s = self._session
        if s is not None:
            return s
        with self._init_lock:
            if self._session is None:
                self._session = StatefulKernelSession(timeout=self._local_timeout)
            return self._session

    def _reply(self, text: str, channel: str | None) -> Message:
        author = Author(role=Role.TOOL, name=self.tool_id())
        msg = Message(author=author, content=[TextContent(text=text)]).with_recipient("assistant")
        return msg.with_channel(channel) if channel else msg

    def process_sync_plus(self, message: Message) -> list[Message]:
        session = self._session_or_create()

        script = ""
        if getattr(message, "content", None):
            first = message.content[0]
            script = getattr(first, "text", "") or ""

        with self._exec_lock:
            try:
                output = session.run(script)
            except TimeoutError as exc:
                output = f"[ERROR] {exc}"
            except Exception as exc:
                output = f"[ERROR] {type(exc).__name__}: {exc}"

        return [self._reply(output, getattr(message, "channel", None))]

    def close(self) -> None:
        s = self._session
        if s is None:
            return
        s.shutdown()
        self._session = None

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()


PythonTool = JupyterPythonTool
LocalJupyterSession = StatefulKernelSession
