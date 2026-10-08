"""End-to-end test with a scripted LLM and the offline MockWeb.

Unlike --mock (random placeholders that the grounding gate rejects), the scripted LLM
returns plausible objects whose quotes really exist in the mock pages, so the *verified*
path is exercised through every stage and the report renderer."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.config import MEDDPICC_ELEMENTS, Settings
from agent.pipeline import run_pipeline
from agent.schemas import (CaseStudyExplanation, ContactDraft, FindingsDraft, GtmRationale, HandoffDraft,
                           LeadProfile, PartnerExtraction, PositioningDraft, QualificationDraft, QueryPlan,
                           RubricBatch, SequenceDraft)

MOCK_QUOTE = "Mock page text about mining operations, autonomous drone docks and inspection in Chile."
EMAIL = open(os.path.join(os.path.dirname(__file__), "..", "input", "lead_sqm.txt"), encoding="utf-8").read()


class ScriptedLLM:
    def __init__(self, settings, log):
        self.log, self.provider, self.model = log, "scripted", "scripted"

    def structured(self, *, system, user, schema, **kw):
        ids = re.findall(r'<source id="(S\d+)"', user)
        name = schema.__name__
        if schema is LeadProfile:
            d = dict(company_legal_name="Sociedad Química y Minera de Chile", company_short_name="SQM",
                     company_name_variants=["SQM"], hq_country="Chile", region="LATAM",
                     region_markets=["Chile", "Peru", "Argentina", "Latin America"], industry="Mining",
                     sub_industry="Lithium", use_case="autonomous drone inspection", asset_or_site_types=["brine sites"],
                     site_count=3, site_locations=["Atacama Desert"], operating_conditions=["24/7", "desert"],
                     stated_pains=["contracted crews expensive and hazardous"], referral_entities=["Anglo American"],
                     timeline_signals=["Q3 budget conversation"], budget_signals=["internal budget conversation"],
                     seniority="VP/Head", matching_keywords=["mining", "inspection", "drone", "dock"])
        elif schema is QualificationDraft:
            els = [dict(element=e, status="unknown", what_we_know="", evidence_quote="", what_is_missing=f"{e} unknown",
                        discovery_questions=[f"Question on {e}?"]) for e in MEDDPICC_ELEMENTS]
            els[MEDDPICC_ELEMENTS.index("Identify Pain")].update(
                status="confirmed", what_we_know="Crews costly and hazardous",
                evidence_quote="Contracted crews are expensive and hazardous")
            fits = [dict(criterion_id="warm_referral", assessment="met", evidence_quote="Referred by Anglo American",
                         note=""),
                    dict(criterion_id="target_industry", assessment="met",
                         evidence_quote="24/7 lithium extraction sites in the Atacama Desert", note="")]
            d = dict(framework_rationale="Enterprise multi-site deal.", elements=els, fit_signals=fits,
                     ambiguities=["Q3 relative to today"], risks=["referral unconfirmed"])
        elif schema is QueryPlan:
            d = {"queries": [{"dimension": dm, "query": f"SQM {dm} {i}", "why": "w"}
                             for dm in ["org_structure", "budget_signals", "news", "stakeholder_letters"] for i in range(2)]}
        elif schema is FindingsDraft:
            dim = re.search(r"dimension='(\w+)'", user).group(1)
            d = {"findings": [dict(dimension=dim, claim=f"claim from {s}", source_id=s, evidence_quote=MOCK_QUOTE,
                                   date_or_period="2026", so_what="matters") for s in ids[:2]]
                 + [dict(dimension=dim, claim="fabricated", source_id=ids[0] if ids else "S1",
                         evidence_quote="This sentence does not appear anywhere in the sources at all.",
                         date_or_period="", so_what="")]}
        elif schema is ContactDraft:
            d = {"summary": "none", "findings": []}
        elif schema is PositioningDraft:
            d = dict(thesis="t", value_hooks=[dict(hook="h", finding_ids=["F01", "F99"], how_to_say_it="x")],
                     likely_stakeholders=[dict(name_or_role="VP Operations", why_involved="owner",
                                               basis="verified", finding_ids=[])],
                     risks_and_objections=["r"], avoid=["a"])
        elif schema is RubricBatch:
            d = {"scores": [dict(source_id=s, industry_fit=4, use_case_fit=4, environment_fit=3, proof_strength=2,
                                 evidence_quote=MOCK_QUOTE, summary="s") for s in ids]}
        elif schema is CaseStudyExplanation:
            d = dict(why_strongest="w", how_to_use_in_sales="h", gaps_vs_lead="g")
        elif schema is PartnerExtraction:
            d = {"partners": [dict(name="Mock Partner One", country="Chile", tier="Gold", types=["System Integrator"],
                                   evidence_quote="Mock Partner One Chile Gold System Integrator"),
                              dict(name="Invented Partner", country="Peru", tier="", types=[],
                                   evidence_quote="Invented Partner Peru Silver Reseller listing line")]}
        elif schema is GtmRationale:
            d = dict(rationale="r", ae_role="a", partner_role="p", risks=[])
        elif schema is SequenceDraft:
            opens = json.loads(re.search(r"must all be targeted\): (\[.*?\])", user).group(1).replace("'", '"'))
            url = re.search(r"<proof_content>\n- (\S+)", user).group(1)
            d = dict(strategy="s", tone_notes="t", reply_branches=[dict(if_reply="yes", next_move="call")],
                     emails=[dict(step=1, send_timing="Day 0", trigger="inbound", goal="g", unknowns_targeted=opens,
                                  finding_ids=["F01"], materials=[url], subject="Re: inspection",
                                  body="Thanks for reaching out. A short body.", why_this_step="first")])
        elif schema is HandoffDraft:
            d = dict(who_he_is="w", what_he_cares_about=["c"],
                     others_involved=[dict(name_or_role="CFO", why_involved="budget", basis="inferred_from_role")],
                     top_findings=[dict(finding_id=f"F0{i}", why_ae_should_care="x") for i in (1, 2, 3)],
                     next_step=dict(action="Book discovery call", owner="AE", timing="this week", rationale="r",
                                    agenda=["a"]),
                     first_call_questions=["q"], watch_outs=["w"])
        else:
            raise AssertionError(f"unscripted schema {name}")
        obj = schema.model_validate(d)
        self.log.llm_call({"schema": name, "response": d})
        return obj


def test_end_to_end_scripted(tmp_path):
    s = Settings()
    s.mock = True
    root = run_pipeline(EMAIL, s, str(tmp_path), llm_factory=ScriptedLLM)
    for f in ["report.html", "report.md", "run_meta.json", "sources.jsonl", "stages/06_handoff.json"]:
        assert os.path.exists(os.path.join(root, f)), f

    research = json.load(open(os.path.join(root, "stages/02_research.json")))
    assert research["findings"], "verified findings expected"
    assert all(f["evidence_quote"] == MOCK_QUOTE for f in research["findings"])
    assert any("quote not found" in r["reason"] for r in research["rejected"])        # fabricated claim removed
    hook = research["positioning"]["value_hooks"][0]
    assert hook["finding_ids"] == ["F01"]                                               # F99 stripped
    assert research["positioning"]["likely_stakeholders"][0]["basis"] == "inferred_from_role"

    qual = json.load(open(os.path.join(root, "stages/01_qualification.json")))
    assert qual["fit_score"] == 30 and len(qual["elements"]) == 8

    gtm = json.load(open(os.path.join(root, "stages/05_gtm.json")))
    assert gtm["partners_found"] == 1 and gtm["partners_rejected"] == 1                 # invented partner rejected
    assert gtm["motion"].startswith("Hybrid")
    assert gtm["shortlist"][0]["name"] == "Mock Partner One"

    seq = json.load(open(os.path.join(root, "stages/03_sequence.json")))
    assert seq["issues"] == [] and seq["revisions"] == 0

    html = open(os.path.join(root, "report.html"), encoding="utf-8").read()
    assert "Stage 6" in html and "Book discovery call" in html and "Mock Partner One" in html
