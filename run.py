"""CLI entry point.

    python run.py input/lead_sqm.txt            # live run (needs API keys in .env)
    python run.py input/lead_sqm.txt --mock     # offline smoke test of the orchestration
"""
import argparse
import os
import sys

from agent.config import Settings
from agent.pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser(description="Inbound lead qualification agent")
    ap.add_argument("input", help="Path to the inbound email (form text or JSON)")
    ap.add_argument("--mock", action="store_true", help="Offline run with fake LLM and web (tests the wiring only)")
    ap.add_argument("--no-cache", action="store_true", help="Ignore cached web fetches/searches")
    ap.add_argument("--out", default="runs", help="Output directory")
    a = ap.parse_args()

    s = Settings()
    s.mock, s.use_cache = a.mock, not a.no_cache
    if not s.mock:
        key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(s.llm_provider)
        local = any(h in s.openai_base_url for h in ("localhost", "127.0.0.1"))
        if key and not os.getenv(key) and not local:
            sys.exit(f"{key} is not set. Copy .env.example to .env and fill it in (or use --mock).")
    with open(a.input, encoding="utf-8") as f:
        raw = f.read()
    run_pipeline(raw, s, a.out)


if __name__ == "__main__":
    main()
