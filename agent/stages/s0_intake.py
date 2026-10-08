"""Stage 0 — intake: parse the raw contact-form text (code) and build a lead profile (LLM)."""
from __future__ import annotations

import json
import re

from ..prompts import PARSE_SYSTEM
from ..schemas import InboundEmail, LeadProfile

FIELDS = {"from": "from", "title": "title", "company": "company", "subject": "subject", "context": "context",
          "message": "context", "body": "context"}


def parse_inbound(raw: str) -> InboundEmail:
    """Accepts 'Field: value' form text (multi-line values allowed) or a JSON object."""
    raw = raw.strip()
    if raw.startswith("{"):
        d = json.loads(raw)
        return InboundEmail(raw=raw, **{k: d.get(k, "") for k in
                                         ["sender_name", "sender_email", "title", "company", "subject", "context"]})
    values: dict[str, str] = {}
    current = None
    for line in raw.splitlines():
        m = re.match(r"^\s*([A-Za-z ]+?)\s*:\s*(.*)$", line)
        key = FIELDS.get(m.group(1).strip().lower()) if m else None
        if key:
            current = key
            values[key] = (values.get(key, "") + " " + m.group(2)).strip()
        elif current:
            values[current] = (values[current] + " " + line.strip()).strip()
    frm = values.get("from", "")
    em = re.search(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", frm)
    name = re.sub(r"\(?[\w.+-]+@[\w.-]+\)?|<.*?>", "", frm).strip(" ()<>,")
    return InboundEmail(sender_name=name, sender_email=em.group(0) if em else "", title=values.get("title", ""),
                        company=values.get("company", ""), subject=values.get("subject", ""),
                        context=values.get("context", ""), raw=raw)


def run(ctx) -> LeadProfile:
    e = ctx.email
    ctx.store.add("inbound://contact-form", f"Inbound email from {e.sender_name}", e.raw, "input", {"email"})
    ctx.log.info(f"parsed: {e.sender_name} <{e.sender_email}> | {e.title} | {e.company}")
    profile = ctx.llm.structured(
        system=PARSE_SYSTEM.format(vendor=ctx.settings.vendor_name, today=ctx.run_date),
        user=f"<email>\n{e.raw}\n</email>\nSender email domain: {e.domain}",
        schema=LeadProfile)
    profile.matching_keywords = [k.lower().strip() for k in profile.matching_keywords if k.strip()]
    ctx.log.info(f"industry={profile.industry} | use_case={profile.use_case} | region={profile.region} | "
                 f"referrals={profile.referral_entities or 'none'}")
    return profile
