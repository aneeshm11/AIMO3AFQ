import argparse
import contextlib
import json
import os
import random
import re
import sys
import threading
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from transformers import AutoTokenizer, set_seed
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    ReasoningEffort,
    RenderConversationConfig,
)

from python_tool import PythonTool


ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def _boxed_int(s: str) -> Optional[int]:
    text = str(s or "")
    for pat in (r"\\boxed\{([^}]*)\}", r"boxed\{([^}]*)\}"):
        hits = re.findall(pat, text)
        for h in reversed(hits):
            h = h.replace(",", " ").strip()
            nums = re.findall(r"-?\d+", h)
            if not nums:
                continue
            try:
                return int(nums[-1])
            except Exception:
                continue
    return None


def _fallback_int(s: str) -> Optional[int]:
    text = str(s or "")
    for pat in (
        r"(?is)final\s+answer\s*(?:is|:)?\s*\$?\s*(-?\d+)",
        r"(?is)the\s+answer\s+is\s*:?\s*\$?\s*(-?\d+)",
        r"(?is)answer\s*[:=]\s*\$?\s*(-?\d+)",
    ):
        hits = re.findall(pat, text)
        for h in reversed(hits):
            try:
                return int(h)
            except Exception:
                continue
    return None


def _predicted_int(full_trace: str) -> int:
    x = _boxed_int(full_trace)
    if x is None:
        x = _fallback_int(full_trace)
    return int(x) if x is not None else 0


def _intish(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if not (x == x):
            return None
        return int(x)
    s = str(x).strip()
    if not s:
        return None
    nums = re.findall(r"-?\d+", s.replace(",", ""))
    if not nums:
        return None
    try:
        return int(nums[-1])
    except Exception:
        return None


def _solution_gt(solution: str) -> Optional[int]:
    sol = str(solution or "")
    b = _boxed_int(sol)
    if b is not None:
        return b
    for pat in (r"answer[:\s]+(-?\d+)", r"answer\s*=\s*(-?\d+)", r"(-?\d+)"):
        hits = re.findall(pat, sol.lower())
        if hits:
            try:
                return int(hits[-1])
            except Exception:
                pass
    return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


@dataclass(frozen=True)
class RunSpec:
    model_path: str
    port: int
    base_url: str
    api_key: str
    tensor_parallel: int
    gpu_mem_util: float
    max_model_len: int
    context_len: int
    max_output_tokens: int
    temperature: float
    top_p: float
    min_p: float
    seed: int
    max_iter: int
    request_timeout_s: float
    cutoff_hours: float
    max_workers: int
    start_index: int
    end_index: int
    out_json: str
    prompt: str
    dataset_name: Optional[str]
    splits: List[str]
    data_file: Optional[str]
    source_values: List[str]
    cuda_visible_devices: Optional[str]


class VllmServer:
    def __init__(self, spec: RunSpec):
        self._spec = spec
        self._proc = None
        self._client = OpenAI(base_url=spec.base_url, api_key=spec.api_key, timeout=spec.request_timeout_s)

    @property
    def client(self) -> OpenAI:
        return self._client

    def ensure_up(self) -> None:
        try:
            self._client.models.list()
            return
        except Exception:
            pass

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self._spec.model_path,
            "--port",
            str(self._spec.port),
            "--tensor-parallel-size",
            str(self._spec.tensor_parallel),
            "--max-model-len",
            str(self._spec.max_model_len),
            "--gpu-memory-utilization",
            str(self._spec.gpu_mem_util),
            "--trust-remote-code",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        t0 = time.time()
        while True:
            try:
                self._client.models.list()
                return
            except Exception:
                if time.time() - t0 > 600:
                    if self._proc and self._proc.poll() is not None:
                        _, stderr = self._proc.communicate()
                        raise RuntimeError(f"vLLM server crashed during startup:\n{stderr.decode(errors='ignore')}")
                    raise RuntimeError("Timed out waiting for vLLM server.")
                time.sleep(5)

    def stop(self) -> None:
        if self._proc is None:
            return
        with contextlib.suppress(Exception):
            self._proc.terminate()
        with contextlib.suppress(Exception):
            self._proc.wait(timeout=30)

    def __del__(self):
        with contextlib.suppress(Exception):
            self.stop()


class TirEngine:
    def __init__(self, spec: RunSpec, server: VllmServer):
        self._spec = spec
        self._server = server
        self._stop_token_ids = ENC.stop_tokens_for_assistant_actions()
        self._tokenizer = None
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(spec.model_path, trust_remote_code=True, fix_mistral_regex=True)
        except Exception:
            self._tokenizer = None

    def _seed_everything(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        set_seed(self._spec.seed)
        random.seed(self._spec.seed)

    def _sys_user_messages(self, user_text: str, pytool: PythonTool) -> List[Message]:
        return [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
                .with_tools(pytool.tool_config),
            ),
            Message.from_role_and_content(Role.USER, user_text),
        ]

    def _render_training(self, msgs: List[Message]) -> str:
        return ENC.decode_utf8(
            ENC.render_conversation_for_training(
                Conversation.from_messages(msgs),
                RenderConversationConfig(auto_drop_analysis=False),
            )
        )

    def solve(self, user_text: str) -> Tuple[str, str, str]:
        self._seed_everything()
        pytool = None
        try:
            pytool = PythonTool(execution_backend="jupyter")
            dialog = self._sys_user_messages(user_text, pytool)
            assistant_only: List[Message] = []
            saw_boxed = False

            for _ in range(int(self._spec.max_iter)):
                prompt_ids = ENC.render_conversation_for_completion(Conversation.from_messages(dialog), Role.ASSISTANT)
                room = int(self._spec.context_len) - len(prompt_ids)
                if room <= 20:
                    break

                max_tokens = min(int(self._spec.max_output_tokens), room)
                token_ids: List[int] = []
                text_acc = ""
                stop_now = False

                try:
                    stream = self._server.client.completions.create(
                        model=self._spec.model_path,
                        prompt=prompt_ids,
                        max_tokens=max_tokens,
                        temperature=float(self._spec.temperature),
                        top_p=float(self._spec.top_p),
                        seed=int(self._spec.seed),
                        stream=True,
                        extra_body=dict(
                            min_p=float(self._spec.min_p),
                            stop_token_ids=self._stop_token_ids,
                            return_token_ids=True,
                        ),
                        timeout=self._spec.request_timeout_s,
                    )

                    for chunk in stream:
                        tchunk = chunk.choices[0].token_ids
                        xchunk = chunk.choices[0].text
                        if tchunk:
                            token_ids.extend(tchunk)
                        if xchunk:
                            text_acc += xchunk
                        if "}" in (xchunk or "") and _boxed_int(text_acc) is not None:
                            saw_boxed = True
                            stop_now = True
                            break

                    with contextlib.suppress(Exception):
                        stream.close()
                except Exception:
                    break

                if token_ids:
                    new_msgs = ENC.parse_messages_from_completion_tokens(token_ids, Role.ASSISTANT)
                    dialog.extend(new_msgs)
                    assistant_only.extend(new_msgs)

                last = dialog[-1] if dialog else None
                if last is None:
                    break

                if getattr(last, "channel", None) == "final" or (token_ids and token_ids[-1] == 200002):
                    break

                if getattr(last, "recipient", None) == "python":
                    tool_reply = pytool.process_sync_plus(last)
                    dialog.extend(tool_reply)

                if stop_now or saw_boxed:
                    break

            full_trace = self._render_training(dialog)
            completion_only = self._render_training(assistant_only) if assistant_only else ""
            input_only = self._render_training(self._sys_user_messages(user_text, pytool))
            return input_only, completion_only, full_trace
        except Exception:
            return "", "", ""
        finally:
            if pytool is not None:
                with contextlib.suppress(Exception):
                    pytool.close()


class ResultLedger:
    def __init__(self, out_path: Path, spec: RunSpec):
        self._out_path = out_path
        self._lock = threading.Lock()
        self._rows: List[Dict[str, Any]] = []
        self._spec = spec
        self._last_flush = 0.0

    def add(self, row: Dict[str, Any]) -> None:
        with self._lock:
            self._rows.append(row)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._rows)

    def flush(self, stats: Dict[str, Any], force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_flush < 30:
            return
        self._last_flush = now
        payload = dict(
            meta=dict(
                model_path=self._spec.model_path,
                base_url=self._spec.base_url,
                start_index=self._spec.start_index,
                end_index=self._spec.end_index,
                cutoff_hours=self._spec.cutoff_hours,
                max_workers=self._spec.max_workers,
                dataset_name=self._spec.dataset_name,
                splits=self._spec.splits,
                data_file=self._spec.data_file,
                source_values=self._spec.source_values,
            ),
            stats=dict(stats),
            results=self.snapshot(),
        )
        write_json(self._out_path, payload)


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        obj = rows

    if isinstance(obj, list):
        rows2 = obj
    elif isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            rows2 = obj["data"]
        else:
            rows2 = [obj]
    else:
        raise ValueError("Unsupported JSON root type")

    out: List[Dict[str, Any]] = []
    for x in rows2:
        if isinstance(x, dict):
            out.append(x)
    return out


def _slice_bounds(n: int, a: int, b: int) -> Tuple[int, int]:
    a2 = max(0, int(a))
    b2 = int(b)
    if b2 < 0:
        b2 = 0
    if b2 > n:
        b2 = n
    if a2 > b2:
        a2 = b2
    return a2, b2


def _load_samples(spec: RunSpec) -> Tuple[List[int], List[Dict[str, Any]]]:
    if spec.data_file:
        rows = _read_json_or_jsonl(Path(spec.data_file))
        if spec.source_values:
            if rows and "source" not in rows[0]:
                print("key not found: source")
            else:
                rows = [r for r in rows if r.get("source") in set(spec.source_values)]
        lo, hi = _slice_bounds(len(rows), spec.start_index, spec.end_index)
        ids = list(range(lo, hi))
        return ids, rows[lo:hi]

    if not spec.dataset_name:
        raise ValueError("Provide either --data-file or --dataset-name")

    from datasets import load_dataset

    all_rows: List[Dict[str, Any]] = []
    for split in spec.splits:
        ds = load_dataset(spec.dataset_name, split=split)
        if spec.source_values:
            if "source" not in getattr(ds, "column_names", []):
                print("key not found: source")
            else:
                keep = set(spec.source_values)
                ds = ds.filter(lambda ex: ex.get("source", None) in keep)
        lo, hi = _slice_bounds(len(ds), spec.start_index, spec.end_index)
        for i in range(lo, hi):
            all_rows.append(ds[i])
    ids = list(range(spec.start_index, spec.start_index + len(all_rows)))
    return ids, all_rows


def _extract_fields(ex: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    if "problem" in ex:
        q = str(ex.get("problem") or "")
        gt = _solution_gt(str(ex.get("solution") or ""))
        return q, gt
    q = str(ex.get("question") or ex.get("problem") or "")
    gt = _intish(ex.get("answer")) if "answer" in ex else None
    return q, gt


def _make_user_text(question: str, prompt: str) -> str:
    return (question or "").strip() + "\n\n" + (prompt or "").strip()


class WorkerPool:
    def __init__(self, n: int, fn):
        self._n = int(n)
        self._fn = fn
        self._q: Queue = Queue(maxsize=max(1, self._n * 2))
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> None:
        for i in range(self._n):
            t = threading.Thread(target=self._loop, name=f"w{i}", daemon=True)
            self._threads.append(t)
            t.start()

    def submit(self, item: Any) -> bool:
        if self._stop.is_set():
            return False
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.25)
                return True
            except Exception:
                continue
        return False

    def stop(self) -> None:
        self._stop.set()
        for _ in self._threads:
            with contextlib.suppress(Exception):
                self._q.put(None, timeout=0.1)

    def join(self) -> None:
        for t in self._threads:
            with contextlib.suppress(Exception):
                t.join()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.25)
            except Empty:
                continue
            if item is None:
                self._q.task_done()
                return
            try:
                self._fn(item)
            except Exception:
                pass
            finally:
                with contextlib.suppress(Exception):
                    self._q.task_done()


def _parse_args() -> RunSpec:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default="sk-local")
    p.add_argument("--tensor-parallel", type=int, default=2)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=100000)
    p.add_argument("--context-len", type=int, default=100000)
    p.add_argument("--max-output-tokens", type=int, default=90000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--min-p", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=60)
    p.add_argument("--request-timeout-s", type=float, default=3600)
    p.add_argument("--cutoff-hours", type=float, default=30.0)
    p.add_argument("--max-workers", type=int, default=64)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=-1)
    p.add_argument("--out-json", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--splits", nargs="+", default=["train"])
    p.add_argument("--data-file", default=None)
    p.add_argument("--source-values", nargs="*", default=[])
    p.add_argument("--cuda-visible-devices", default=None)
    a = p.parse_args()

    base_url = a.base_url or f"http://127.0.0.1:{int(a.port)}/v1"
    if int(a.end_index) <= int(a.start_index):
        raise ValueError("--end-index must be > --start-index")

    return RunSpec(
        model_path=a.model_path,
        port=int(a.port),
        base_url=str(base_url),
        api_key=str(a.api_key),
        tensor_parallel=int(a.tensor_parallel),
        gpu_mem_util=float(a.gpu_mem_util),
        max_model_len=int(a.max_model_len),
        context_len=int(a.context_len),
        max_output_tokens=int(a.max_output_tokens),
        temperature=float(a.temperature),
        top_p=float(a.top_p),
        min_p=float(a.min_p),
        seed=int(a.seed),
        max_iter=int(a.max_iter),
        request_timeout_s=float(a.request_timeout_s),
        cutoff_hours=float(a.cutoff_hours),
        max_workers=int(a.max_workers),
        start_index=int(a.start_index),
        end_index=int(a.end_index),
        out_json=str(a.out_json),
        prompt=str(a.prompt),
        dataset_name=a.dataset_name,
        splits=list(a.splits or []),
        data_file=a.data_file,
        source_values=list(a.source_values or []),
        cuda_visible_devices=a.cuda_visible_devices,
    )


def main() -> None:
    spec = _parse_args()
    if spec.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.cuda_visible_devices)

    ids, samples = _load_samples(spec)
    if not samples:
        write_json(Path(spec.out_json), dict(meta=dict(), stats=dict(processed=0), results=[]))
        return

    server = VllmServer(spec)
    server.ensure_up()
    engine = TirEngine(spec, server)

    out_path = Path(spec.out_json)
    ledger = ResultLedger(out_path, spec)

    stats_lock = threading.Lock()
    stats = dict(processed=0, correct=0, errors=0, started_at=time.time(), deadline_hit=False)

    deadline = float("inf")
    if spec.cutoff_hours and spec.cutoff_hours > 0:
        deadline = time.time() + float(spec.cutoff_hours) * 3600.0

    stop_submissions = threading.Event()

    def handle_one(job: Tuple[int, Dict[str, Any]]) -> None:
        pid, ex = job
        t0 = time.time()
        q, gt = _extract_fields(ex)
        user_text = _make_user_text(q, spec.prompt)
        row: Dict[str, Any] = dict(
            problem_id=int(pid),
            raw_question=q,
            raw_prompt=spec.prompt,
            ground_truth_answer=gt,
        )
        try:
            chat_in, completion_only, full_trace = engine.solve(user_text)
            pred = _predicted_int(full_trace)
            row.update(
                chat_templated_prompt_question=chat_in,
                solution_text=completion_only,
                chat_templated_full_trace=full_trace,
                predicted_answer=pred,
                is_correct=bool(gt is not None and int(pred) == int(gt)),
                wall_time_s=round(time.time() - t0, 6),
            )
            with stats_lock:
                stats["processed"] += 1
                if row["is_correct"]:
                    stats["correct"] += 1
        except Exception as e:
            row.update(error=f"{type(e).__name__}: {e}", wall_time_s=round(time.time() - t0, 6))
            with stats_lock:
                stats["processed"] += 1
                stats["errors"] += 1
        ledger.add(row)
        with stats_lock:
            ledger.flush(stats, force=False)

    pool = WorkerPool(spec.max_workers, handle_one)
    pool.start()

    try:
        for pid, ex in zip(ids, samples):
            if time.time() >= deadline:
                with stats_lock:
                    stats["deadline_hit"] = True
                stop_submissions.set()
                break
            ok = pool.submit((pid, ex))
            if not ok:
                break

        if stop_submissions.is_set():
            with stats_lock:
                ledger.flush(stats, force=True)

        pool.stop()
        pool.join()
        with stats_lock:
            stats["finished_at"] = time.time()
            ledger.flush(stats, force=True)
    finally:
        with contextlib.suppress(Exception):
            server.stop()


if __name__ == "__main__":
    main()
