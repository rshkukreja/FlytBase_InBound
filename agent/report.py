"""Renders the run into report.html (for reading) and report.md (for diffing / pasting)."""
from __future__ import annotations

import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES = os.path.join(os.path.dirname(__file__), "templates")
DIM_LABELS = {"org_structure": "Organizational structure & reporting lines",
              "budget_signals": "Budget signals (filings, capex, technology spend)",
              "news": "Recent news & strategic announcements",
              "stakeholder_letters": "Stakeholder / investor communications",
              "contact": "Sender verification"}


def render(ctx, meta: dict, out_dir: str) -> tuple[str, str]:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]),
                      trim_blocks=True, lstrip_blocks=True)
    r = ctx.out["research"]
    data = dict(
        meta=meta, email=ctx.email, profile=ctx.out["profile"], q=ctx.out["qualification"], r=r,
        cs=ctx.out["case_study"], gtm=ctx.out["gtm"], seq=ctx.out["sequence"], h=ctx.out["handoff"],
        src={s.id: s for s in r.sources}, all_sources=[s for s in ctx.store.all()],
        dims=DIM_LABELS, findings_by_dim={d: [f for f in r.findings if f.dimension == d] for d in DIM_LABELS},
        finding={f.id: f for f in r.findings}, vendor=ctx.settings.vendor_name,
    )
    html_path = os.path.join(out_dir, "report.html")
    md_path = os.path.join(out_dir, "report.md")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(env.get_template("report.html.j2").render(**data))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(env.get_template("report.md.j2").render(**data))
    return html_path, md_path
