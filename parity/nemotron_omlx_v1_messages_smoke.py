"""End-to-end Anthropic ``POST /v1/messages`` smoke test for the oMLX + quanta shim (task #162).

Spins up the quanta-patched oMLX server pointed at a single-model directory (an isolated symlink
to the resident Nemotron int4-g64 artifact), hits ``POST /v1/messages`` twice, and asserts the
response is a valid Anthropic-shape JSON with non-empty ``content[0].text``.

Gates the integration layer end-to-end:

  1. ``quanta-omlx`` console-script handoff to oMLX's CLI
     (:func:`quanta.omlx_patch.main` → arms the autopatch → ``omlx.cli.main()``).
  2. Lazy ``omlx.model_discovery`` + ``omlx.engine_pool`` patches — the artifact is registered
     with ``engine_type='quanta'`` and routed to :func:`quanta.shim.omlx.load_quanta_engine`.
  3. :class:`quanta.shim.omlx.QuantaOmlxEngine` surface required by the Anthropic /v1/messages
     handler:
       * :meth:`~QuantaOmlxEngine.count_chat_tokens` — input-token accounting / context-window
         validation. Without this method every request 500s with
         ``AttributeError: 'QuantaOmlxEngine' object has no attribute 'count_chat_tokens'``
         (the gap this smoke test surfaced + the patch that resolved it).
       * :meth:`~QuantaOmlxEngine.stream_generate` — already covered by /v1/chat/completions
         smokes, re-exercised here under the Anthropic protocol's slightly different chat-
         template + stop-reason mapping.
  4. Round-trip Anthropic-shape JSON serialization (id/type/role/content/stop_reason/usage)
     via :func:`omlx.api.anthropic_utils.convert_internal_to_anthropic_response`.

ORCHESTRATOR runs this — do NOT invoke from an agent. Loads ~68 GB resident Nemotron int4-g64.
Default port is **8000** (oMLX default); override with ``--port``.

Run:  ``uv run python -m parity.nemotron_omlx_v1_messages_smoke``
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def _port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def _wait_ready(host: str, port: int, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1.0) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
            pass
        time.sleep(0.5)
    return False


def _post_messages(host: str, port: int, model: str, prompt: str, max_tokens: int,
                   timeout_s: float = 600.0) -> dict:
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/messages",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.read().decode('utf-8', errors='replace')[:500]}")
        return json.loads(r.read())


def _assert_anthropic_shape(resp: dict, model_id: str) -> None:
    """Strict shape check: every Anthropic Messages-API response field present + sane."""
    for k in ("id", "type", "role", "model", "content", "stop_reason", "usage"):
        if k not in resp:
            raise AssertionError(f"missing field {k!r} in response: {resp!r}")
    if resp["type"] != "message":
        raise AssertionError(f"expected type='message', got {resp['type']!r}")
    if resp["role"] != "assistant":
        raise AssertionError(f"expected role='assistant', got {resp['role']!r}")
    if resp["model"] != model_id:
        raise AssertionError(f"expected model={model_id!r}, got {resp['model']!r}")
    content = resp["content"]
    if not isinstance(content, list) or not content:
        raise AssertionError(f"content must be a non-empty list, got {content!r}")
    first = content[0]
    if first.get("type") != "text":
        raise AssertionError(f"content[0].type expected 'text', got {first!r}")
    if not isinstance(first.get("text"), str) or not first["text"]:
        raise AssertionError(f"content[0].text empty: {first!r}")
    u = resp["usage"]
    if not (isinstance(u.get("input_tokens"), int) and u["input_tokens"] > 0):
        raise AssertionError(f"usage.input_tokens invalid: {u!r}")
    if not (isinstance(u.get("output_tokens"), int) and u["output_tokens"] > 0):
        raise AssertionError(f"usage.output_tokens invalid: {u!r}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--artifact",
                   default=str(Path.home() / "models" / "NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"),
                   help="absolute path to the quanta-baked artifact directory")
    p.add_argument("--model-id", default="nemotron-int4g64",
                   help="alias for the artifact inside the staged --model-dir (= oMLX model_id)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--ready-timeout", type=float, default=30.0)
    p.add_argument("--first-call-timeout", type=float, default=600.0,
                   help="first /v1/messages call triggers the ~68GB model load")
    args = p.parse_args()

    art = Path(args.artifact).expanduser()
    if not art.is_dir():
        print(f"FAIL — artifact not found: {art}", file=sys.stderr)
        return 2
    if not _port_free(args.port, args.host):
        print(f"FAIL — port {args.host}:{args.port} busy (stop any other oMLX server first)",
              file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="quanta_omlx_smoke_") as tmp:
        model_dir = Path(tmp)
        (model_dir / args.model_id).symlink_to(art)
        cmd = ["uv", "run", "quanta-omlx", "serve",
               "--model-dir", str(model_dir),
               "--host", args.host, "--port", str(args.port),
               "--log-level", "info",
               "--max-model-memory", "disabled",
               "--no-cache"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            if not _wait_ready(args.host, args.port, timeout_s=args.ready_timeout):
                print(f"FAIL — server not ready on {args.host}:{args.port} "
                      f"within {args.ready_timeout}s", file=sys.stderr)
                return 2

            t0 = time.monotonic()
            r1 = _post_messages(args.host, args.port, args.model_id,
                                "Say hello in five words.", args.max_tokens,
                                timeout_s=args.first_call_timeout)
            cold_s = time.monotonic() - t0
            _assert_anthropic_shape(r1, args.model_id)
            print(f"[OK] cold /v1/messages: {cold_s:.1f}s  "
                  f"in={r1['usage']['input_tokens']} out={r1['usage']['output_tokens']} "
                  f"stop={r1['stop_reason']}  text[:80]={r1['content'][0]['text'][:80]!r}")

            t1 = time.monotonic()
            r2 = _post_messages(args.host, args.port, args.model_id,
                                "What is 2+2?", min(16, args.max_tokens), timeout_s=60.0)
            warm_s = time.monotonic() - t1
            _assert_anthropic_shape(r2, args.model_id)
            print(f"[OK] warm /v1/messages: {warm_s:.2f}s  "
                  f"in={r2['usage']['input_tokens']} out={r2['usage']['output_tokens']} "
                  f"text[:80]={r2['content'][0]['text'][:80]!r}")
            print("\n/v1/messages smoke OK — Anthropic-shape JSON, non-empty content[0].text")
            return 0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
