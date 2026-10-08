"""Orchestrator. Stage numbering follows the brief; execution order follows data
dependencies: the email sequence (Stage 3) needs the case study (4) and GTM motion (5)."""
from __future__ import annotations

import datetime as dt
import json
import os
import time

from .config import RunPaths, Settings
from .context import Ctx
from .evidence import EvidenceStore
from .llm import LLM
from .report import render
from .runlog import RunLog, console
from .stages import s0_intake, s1_qualify, s2_research, s3_sequence, s4_case_study, s5_partner, s6_handoff
from .tools.web import MockWeb, Web

PLAN = [
    ("profile", "Stage 0 · Intake: parse email, build lead profile", s0_intake.run, "00_lead_profile.json"),
    ("qualification", "Stage 1 · Qualification (MEDDPICC + ICP fit)", s1_qualify.run, "01_qualification.json"),
    ("research", "Stage 2 · Deep account research (grounded)", s2_research.run, "02_research.json"),
    ("case_study", "Stage 4 · Case study & material matching", s4_case_study.run, "04_case_study.json"),
    ("gtm", "Stage 5 · Partner identification & GTM motion", s5_partner.run, "05_gtm.json"),
    ("sequence", "Stage 3 · Adaptive response sequence", s3_sequence.run, "03_sequence.json"),
    ("handoff", "Stage 6 · AE handoff summary", s6_handoff.run, "06_handoff.json"),
]


def run_pipeline(raw_email: str, settings: Settings, out_dir: str = "runs", llm_factory=None, web_factory=None) -> str:
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3] + ("-mock" if settings.mock else "")
    paths = RunPaths(os.path.join(out_dir, run_id))
    log = RunLog(paths)
    llm = (llm_factory or LLM)(settings, log)
    web = (web_factory or (MockWeb if settings.mock else Web))(settings, log)
    email = s0_intake.parse_inbound(raw_email)
    ctx = Ctx(settings=settings, llm=llm, web=web, store=EvidenceStore(), log=log, email=email,
              run_date=dt.date.today().isoformat())
    with open(os.path.join(paths.root, "input_email.txt"), "w", encoding="utf-8") as f:
        f.write(raw_email)
    console.print(f"[bold]Run {run_id}[/bold] · LLM {llm.provider}:{llm.model} · "
                  f"search {'tavily' if settings.tavily_key else 'duckduckgo'} · output {paths.root}")
    t0 = time.time()
    for key, title, fn, fname in PLAN:
        with log.stage(key, title):
            ctx.out[key] = fn(ctx)
            log.save_stage(fname, ctx.out[key])
    # full evidence (source texts) for audit
    with open(os.path.join(paths.root, "sources.jsonl"), "w", encoding="utf-8") as f:
        for s in ctx.store.all():
            f.write(json.dumps({"id": s.id, "url": s.url, "title": s.title, "kind": s.kind, "tags": sorted(s.tags),
                                "chars": len(s.text), "text": s.text[:200000]}, ensure_ascii=False) + "\n")
    meta = {"run_id": run_id, "run_date": ctx.run_date, "seconds": round(time.time() - t0, 1),
            "llm": f"{llm.provider}:{llm.model}", "search": "tavily" if settings.tavily_key else "duckduckgo",
            "mock": settings.mock, "stages": log.stage_stats, "counters": log.counters}
    with open(os.path.join(paths.root, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    html_path, md_path = render(ctx, meta, paths.root)
    console.rule("[bold green]Done")
    console.print(f"Report: {html_path}\nMarkdown: {md_path}\nStage JSON + LLM call logs: {paths.root}")
    return paths.root
