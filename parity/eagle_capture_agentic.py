"""On-policy, chat-format agentic-coding capture for the EAGLE-3 drafter (runs alone — 389 GiB).

The decode distribution the drafter must accelerate is the **target's own generations in its chat
format** — not raw repo files. For each agentic-coding prompt we render the Kimi chat template
(``apply_chat_template``), let the *target* generate its continuation (sampled — this is a reasoning
model that loops under greedy), then teacher-force [prompt + generation] back through the target to
capture (low/mid/high fused hidden, input token, target's argmax next token). Each conversation is
captured independently (clean per-sequence context).

    uv run --with 'tiktoken' --with 'jinja2' python -m parity.eagle_capture_agentic [n_prompts] [gen_len]
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

from quanta.eagle.capture import capture_features, save_features
from quanta.generate import generate
from quanta.runtime import ResidentModel
from quanta.tokenizer import KimiTokenizer

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int2g64"
OUT = "/Users/pmrj/models/kimi_eagle/features_agentic.safetensors"
LAYERS = (10, 30, 50)
SYSTEM = "You are an expert software engineer. Be concise and write correct, runnable code."

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number using memoization.",
    "Implement binary search over a sorted list in Python and explain the invariant.",
    "Refactor this into a list comprehension: result=[]\nfor x in xs:\n    if x%2==0:\n        result.append(x*x)",
    "Write a Python decorator that retries a function up to 3 times on exception with exponential backoff.",
    "Find and fix the bug:\ndef avg(xs):\n    return sum(xs)/len(xs)\nprint(avg([]))",
    "Write a dataclass for a 2D point with a method computing Euclidean distance to another point.",
    "Implement an LRU cache in Python without functools, with O(1) get and put.",
    "Write a regex to extract all IPv4 addresses from a string, in Python.",
    "Explain what this does and rewrite it more clearly:\nf=lambda n:n<2 or n%2 and f(n-2)",
    "Write a Python context manager that times the enclosed block and prints the elapsed ms.",
    "Implement merge sort in Python and give its time and space complexity.",
    "Write a function to flatten an arbitrarily nested list of integers in Python.",
    "Add type hints and a docstring to:\ndef parse(s):\n    return {k:int(v) for k,v in (p.split('=') for p in s.split('&'))}",
    "Write a SQL query to find the top 3 customers by total order value from orders(customer_id, amount).",
    "Write a TypeScript function that debounces an async callback with a configurable delay.",
    "Implement a thread-safe counter in Python using a lock; show example usage.",
    "Write a Rust function that returns the maximum subarray sum (Kadane's algorithm).",
    "Convert this callback code to async/await in JavaScript:\nfs.readFile('a',(e,d)=>cb(e,d))",
    "Write a bash one-liner to find the 5 largest files under the current directory.",
    "Implement a simple stack-based calculator for + - * / over space-separated tokens, in Python.",
    "Write pytest tests for a function add(a,b) including edge cases.",
    "Explain the difference between a process and a thread, with a Python example of each.",
    "Write a Python generator that yields prime numbers indefinitely.",
    "Given a function with an off-by-one bug in a loop range, identify it:\nfor i in range(len(a)-1):\n    print(a[i+1])",
    "Write a function to compute the edit distance between two strings (dynamic programming), in Python.",
    "Implement a minimal HTTP GET using only the socket module in Python.",
    "Write a Go function that reverses a slice in place.",
    "Refactor a long if/elif chain that maps status codes to messages into a dict lookup, in Python.",
    "Write a Python function to safely read a JSON file, returning a default on any error.",
    "Implement quicksort in Python with a randomized pivot.",
    "Write a function that groups a list of dicts by a given key, in Python.",
    "Explain and fix a deadlock risk in two threads each acquiring two locks in opposite order.",
    "Write a Python script that watches a directory and prints new file names as they appear.",
    "Implement a trie (prefix tree) in Python with insert and startswith.",
    "Write a function to validate balanced parentheses/brackets/braces in a string, in Python.",
    "Given a NumPy array, write code to normalize each row to unit L2 norm.",
]


def run() -> None:
    n_prompts = int(sys.argv[1]) if len(sys.argv) > 1 else len(PROMPTS)
    gen_len = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    mx.set_wired_limit(int(490 * 1024**3))
    t0 = time.perf_counter()
    rm = ResidentModel(ART)
    tok = KimiTokenizer(ART, bos_id=rm.cfg.bos_token_id)

    all_f, all_i, all_t, total_gen = [], [], [], 0
    for k, prompt in enumerate(PROMPTS[:n_prompts]):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        pids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        gen = generate(rm, pids, max_new_tokens=gen_len, temperature=0.7, top_p=0.9,
                       eos_id=tok.eos_id, sparse=None)
        seq = list(pids) + gen
        f, i, t = capture_features(rm, seq, LAYERS, chunk=4096)
        all_f.append(f)
        all_i.append(i)
        all_t.append(t)
        total_gen += len(gen)
        print(f"  [{k + 1}/{n_prompts}] prompt {len(pids)} + gen {len(gen)} tok "
              f"({total_gen} gen so far, {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    feat3 = mx.concatenate(all_f, 0)
    save_features(OUT, feat3, mx.concatenate(all_i, 0), mx.concatenate(all_t, 0), LAYERS)
    print(f"\nsaved {OUT}: {feat3.shape[0]} tokens ({total_gen} generated) in "
          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    run()
