"""Run recorder: every LLM call, web call and stage output is written to disk so the
whole run can be inspected after the fact (runs/<run_id>/...)."""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager

from rich.console import Console

console = Console()


class RunLog:
    def __init__(self, paths):
        self.paths = paths
        self._lock = threading.Lock()
        self._n_llm = 0
        self.events_path = os.path.join(paths.root, "run_log.jsonl")
        self.stage_stats: list[dict] = []
        self.counters = {"llm_calls": 0, "llm_input_tokens": 0, "llm_output_tokens": 0,
                         "searches": 0, "fetches": 0, "fetch_failures": 0, "cache_hits": 0}
        self._current_stage = "setup"

    # ---- generic events -------------------------------------------------
    def event(self, kind: str, **data):
        rec = {"t": round(time.time(), 3), "stage": self._current_stage, "kind": kind, **data}
        with self._lock:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def bump(self, key: str, n: int = 1):
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + n

    def info(self, msg: str):
        console.print(f"   [dim]{msg}[/dim]")
        self.event("info", msg=msg)

    def warn(self, msg: str):
        console.print(f"   [yellow]! {msg}[/yellow]")
        self.event("warning", msg=msg)

    # ---- LLM calls --------------------------------------------------------
    def llm_call(self, record: dict):
        with self._lock:
            self._n_llm += 1
            n = self._n_llm
        name = f"{n:03d}_{self._current_stage}_{record.get('schema', 'x')}.json"
        with open(os.path.join(self.paths.llm_dir, name), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)
        self.bump("llm_calls")
        usage = record.get("usage") or {}
        self.bump("llm_input_tokens", int(usage.get("input_tokens") or 0))
        self.bump("llm_output_tokens", int(usage.get("output_tokens") or 0))
        self.event("llm_call", file=name, schema=record.get("schema"), seconds=record.get("seconds"))

    # ---- stages -------------------------------------------------------------
    @contextmanager
    def stage(self, key: str, title: str):
        self._current_stage = key
        before = dict(self.counters)
        console.rule(f"[bold cyan]{title}")
        t0 = time.time()
        status = "ok"
        try:
            yield
        except Exception as e:  # recorded, then re-raised
            status = f"error: {e}"
            raise
        finally:
            secs = round(time.time() - t0, 1)
            delta = {k: self.counters.get(k, 0) - before.get(k, 0) for k in self.counters}
            self.stage_stats.append({"stage": key, "title": title, "seconds": secs, "status": status, **delta})
            self.event("stage_done", seconds=secs, status=status, **delta)
            console.print(f"   [green]done in {secs}s[/green] "
                          f"[dim](llm {delta['llm_calls']}, search {delta['searches']}, fetch {delta['fetches']})[/dim]")

    def save_stage(self, filename: str, model) -> str:
        path = os.path.join(self.paths.stage_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            if hasattr(model, "model_dump_json"):
                f.write(model.model_dump_json(indent=2))
            else:
                json.dump(model, f, ensure_ascii=False, indent=2, default=str)
        return path
