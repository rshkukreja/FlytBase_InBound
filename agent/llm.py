"""Structured LLM calls.

Every call asks for ONE Pydantic schema. With Anthropic we force a tool call whose
input_schema is that schema; with OpenAI we use JSON mode. Output is validated with
Pydantic and, on failure, re-asked with the validation error (max 3 attempts).
Each call (prompt + response + usage) is written to runs/<id>/llm_calls/.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def inline_refs(schema: dict) -> dict:
    """Pydantic emits $defs/$ref; inline them so any provider accepts the schema."""
    defs = schema.get("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                merged = {**defs[name], **{k: v for k, v in node.items() if k != "$ref"}}
                return resolve(merged)
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)


class LLM:
    def __init__(self, settings, log):
        self.s = settings
        self.log = log
        self.provider = "mock" if settings.mock else settings.llm_provider
        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
            self.model = settings.anthropic_model
        elif self.provider == "openai":
            # Any OpenAI-compatible endpoint: OpenAI, Google Gemini, Groq, OpenRouter, local Ollama...
            import openai
            if not settings.openai_model:
                raise SystemExit("OPENAI_MODEL must be set when LLM_PROVIDER=openai")
            self.client = openai.OpenAI(base_url=settings.openai_base_url or None,
                                        api_key=os.getenv("OPENAI_API_KEY") or "not-needed")
            self.model = settings.openai_model
        else:
            self.client = None
            self.model = "mock"
        self._rf_mode = settings.openai_response_format  # json_schema | json_object | none
        self._no_reasoning_param = False
        self._pace_lock = threading.Lock()
        self._last_call = 0.0

    def _pace(self):
        """Free tiers limit requests per minute: keep a minimum gap between calls."""
        gap = self.s.llm_min_interval
        if gap <= 0:
            return
        with self._pace_lock:
            wait = self._last_call + gap - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    # ------------------------------------------------------------------
    def structured(self, *, system: str, user: str, schema: type[T], max_tokens: int = 8000,
                   temperature: float = 0.2) -> T:
        tool_schema = inline_refs(schema.model_json_schema())
        required = tool_schema.get("required", [])
        feedback = ""
        last_err = None
        for attempt in range(1, 5):
            prompt = user if not feedback else (
                user + "\n\n<previous_attempt_error>\n" + feedback
                + "\nReturn ONE complete JSON object with all required top-level fields.\n</previous_attempt_error>")
            t0 = time.time()
            raw, usage, meta = self._call(system, prompt, schema.__name__, tool_schema, max_tokens, temperature)
            secs = round(time.time() - t0, 2)
            record = {"schema": schema.__name__, "provider": self.provider, "model": self.model, "attempt": attempt,
                      "seconds": secs, "usage": usage, "system": system, "user": prompt, "response": raw, **meta}
            try:
                obj = schema.model_validate(raw)
                self.log.llm_call({**record, "valid": True})
                return obj
            except ValidationError as e:
                last_err = e
                missing_top = [k for k in required if not isinstance(raw, dict) or k not in raw]
                if meta.get("finish_reason") == "length":
                    feedback = ("Your previous reply was cut off before the JSON was complete. "
                                "Keep every text field short (one or two sentences) so the whole object fits.")
                elif not raw:
                    feedback = "Your previous reply was not valid JSON. Output only the JSON object, nothing else."
                elif len(missing_top) == len(required) and required:
                    feedback = (f"Your previous reply was not the requested top-level object. The top-level object "
                                f"must have these fields: {required}.")
                else:
                    feedback = "Your previous output failed validation:\n" + str(e)[:2500]
                self.log.llm_call({**record, "valid": False, "error": feedback})
                self.log.warn(f"{schema.__name__} invalid (attempt {attempt}): {feedback[:90]}; retrying")
        raise RuntimeError(f"LLM could not produce a valid {schema.__name__}: {last_err}")

    # ------------------------------------------------------------------
    def _call(self, system, user, name, tool_schema, max_tokens, temperature):
        if self.provider == "anthropic":
            resp = self._with_retries(lambda: self.client.messages.create(
                model=self.model, max_tokens=max_tokens, temperature=temperature, system=system,
                tools=[{"name": "submit_result",
                        "description": f"Submit the {name} object. Always call this tool.",
                        "input_schema": tool_schema}],
                tool_choice={"type": "tool", "name": "submit_result"},
                messages=[{"role": "user", "content": user}],
            ))
            usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
            meta = {"finish_reason": resp.stop_reason}
            for block in resp.content:
                if getattr(block, "type", "") == "tool_use":
                    return block.input, usage, meta
            return {}, usage, meta
        if self.provider == "openai":
            sys = (system + "\n\nRespond ONLY with one JSON object (no prose, no code fences) that validates "
                   "against this JSON Schema:\n" + json.dumps(tool_schema))
            msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
            strict_schema = clean_schema(tool_schema)

            def call():
                # Generous output budget: "thinking" models spend part of it on reasoning,
                # and a truncated reply is invalid JSON.
                kw = {"model": self.model, "messages": msgs, "temperature": temperature,
                      "max_tokens": max(max_tokens, self.s.openai_min_output_tokens)}
                if self.s.reasoning_effort and not self._no_reasoning_param:
                    kw["reasoning_effort"] = self.s.reasoning_effort
                # Strongest format the endpoint accepts: schema-enforced JSON -> JSON mode -> plain text.
                while True:
                    extra = {}
                    if self._rf_mode == "json_schema":
                        extra["response_format"] = {"type": "json_schema", "json_schema": {
                            "name": name, "schema": strict_schema}}
                    elif self._rf_mode == "json_object":
                        extra["response_format"] = {"type": "json_object"}
                    try:
                        return self.client.chat.completions.create(**kw, **extra)
                    except Exception as e:
                        msg = str(e)
                        is_bad_request = getattr(e, "status_code", None) == 400 or "400" in msg[:40]
                        if is_bad_request and "reasoning" in msg.lower() and "reasoning_effort" in kw:
                            self._no_reasoning_param = True
                            kw.pop("reasoning_effort")
                            self.log.warn("endpoint rejected reasoning_effort; continuing without it")
                            continue
                        if is_bad_request and self._rf_mode != "none":
                            nxt = {"json_schema": "json_object", "json_object": "none"}[self._rf_mode]
                            self.log.warn(f"endpoint rejected {self._rf_mode} output ({msg[:100]}); trying {nxt}")
                            self._rf_mode = nxt
                            continue
                        raise

            resp = self._with_retries(call)
            u = getattr(resp, "usage", None)
            usage = {"input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                     "output_tokens": getattr(u, "completion_tokens", 0) or 0}
            text = resp.choices[0].message.content or ""
            meta = {"finish_reason": resp.choices[0].finish_reason, "format": self._rf_mode,
                    "raw_text": text[:20000]}
            return extract_json(text), usage, meta
        return SchemaFaker().make(tool_schema), {"input_tokens": 0, "output_tokens": 0}, {}

    def _with_retries(self, fn, tries: int = 6):
        for i in range(tries):
            self._pace()
            try:
                return fn()
            except Exception as e:  # rate limits / transient network
                if i == tries - 1:
                    raise
                rate_limited = "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower()
                wait = (20 * (i + 1) if rate_limited else 2 ** i) + random.random()
                self.log.warn(f"LLM API error ({type(e).__name__}: {str(e)[:120]}); retry in {wait:.1f}s")
                time.sleep(wait)


def extract_json(text: str) -> dict:
    """Parse the model's JSON object. Handles code fences, surrounding prose, and truncated or
    slightly malformed JSON (via json_repair). Never returns an inner fragment: only the outermost
    object counts, so a cut-off reply cannot be mistaken for a nested item."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    try:
        v = json.loads(t)
        if isinstance(v, dict):
            return v
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start == -1:
        return {}
    try:
        from json_repair import repair_json
        v = repair_json(t[start:], return_objects=True)
        if isinstance(v, list):  # e.g. repaired '{...} trailing text' -> take the first object
            v = next((x for x in v if isinstance(x, dict)), {})
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def clean_schema(schema: dict) -> dict:
    """Schema for providers' native structured output: drop keywords some reject (title, default)."""
    if isinstance(schema, dict):
        return {k: clean_schema(v) for k, v in schema.items() if k not in ("title", "default")
                or (k == "title" and not isinstance(v, str))}
    if isinstance(schema, list):
        return [clean_schema(x) for x in schema]
    return schema


class SchemaFaker:
    """Produces schema-valid placeholder objects. Used only by --mock to smoke-test
    the orchestration offline; mock output is never a real result."""

    def make(self, schema: dict, key: str = ""):
        if "anyOf" in schema:
            non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
            return self.make(non_null[0], key) if non_null else None
        if "enum" in schema:
            return schema["enum"][0]
        t = schema.get("type")
        if t == "object":
            return {k: self.make(v, k) for k, v in schema.get("properties", {}).items()}
        if t == "array":
            return [self.make(schema.get("items", {}), key) for _ in range(2)]
        if t == "integer":
            return max(schema.get("minimum", 0), min(schema.get("maximum", 3), 3))
        if t == "number":
            return 1.0
        if t == "boolean":
            return False
        if key.endswith("_id") or key == "source_id":
            return "S1"
        return f"[mock {key or 'text'}]"
