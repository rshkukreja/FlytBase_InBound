"""Unit tests for the deterministic parts of the agent (no network, no LLM)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.evidence import EvidenceStore, verify_quote
from agent.schemas import (ElementAssessment, FitSignal, QualificationDraft, SequenceDraft, SequenceEmail)
from agent.stages.s0_intake import parse_inbound
from agent.stages.s1_qualify import score
from agent.stages.s3_sequence import check
from agent.stages.s5_partner import decide
from agent.tools.site import _locs, is_default_language, section_of

EMAIL = open(os.path.join(os.path.dirname(__file__), "..", "input", "lead_sqm.txt"), encoding="utf-8").read()


# ---------------------------------------------------------------- intake
def test_parse_inbound():
    e = parse_inbound(EMAIL)
    assert e.sender_name == "Rodrigo Castillo"
    assert e.sender_email == "r.castillo@sqm.cl"
    assert e.domain == "sqm.cl"
    assert e.title.startswith("Head of Operations")
    assert "Referred by Anglo American" in e.context


# ---------------------------------------------------------------- grounding gate
TEXT = ("The Company expects capital expenditures of approximately US$1,100 million in 2026,\n"
        "mainly related to lithium expansion projects in the Salar de Atacama and Antofagasta.")


def test_quote_exact_and_whitespace():
    assert verify_quote("capital expenditures of approximately US$1,100 million in 2026", TEXT)[0]
    assert verify_quote("in 2026,   mainly related to lithium expansion", TEXT)[0]


def test_quote_curly_quotes_and_case():
    assert verify_quote("THE COMPANY EXPECTS capital expenditures", TEXT)[0]


def test_quote_fabricated_rejected():
    assert not verify_quote("capital expenditures of approximately US$2,500 million for drones", TEXT)[0]
    assert not verify_quote("The company plans to buy 300 inspection drones in 2026", TEXT)[0]


def test_quote_too_short_rejected():
    assert not verify_quote("lithium", TEXT)[0]


def test_quote_ellipsis_parts():
    assert verify_quote("The Company expects capital expenditures ... lithium expansion projects in the Salar", TEXT)[0]


def test_excerpt_keyword_windows():
    store = EvidenceStore()
    body = ("filler line\n" * 2000) + "Capital expenditure guidance is US$900 million.\n" + ("more filler\n" * 2000)
    s = store.add("http://x", "x", body, "filing", {"budget_signals"})
    ex = store.excerpt(s.id, ["capital expenditure"], 3000)
    assert "US$900 million" in ex and len(ex) <= 3100


# ---------------------------------------------------------------- qualification scoring
def _draft(elements, fits):
    return QualificationDraft(framework_rationale="x", elements=elements, fit_signals=fits, ambiguities=[], risks=[])


def test_unverified_quote_is_downgraded():
    els = [ElementAssessment(element="Identify Pain", status="confirmed", what_we_know="crews costly",
                             evidence_quote="Contracted crews are expensive and hazardous", what_is_missing="",
                             discovery_questions=[]),
           ElementAssessment(element="Economic Buyer", status="confirmed", what_we_know="CFO approved",
                             evidence_quote="The CFO has approved a budget of 2 million", what_is_missing="",
                             discovery_questions=[])]
    q = score(_draft(els, []), EMAIL)
    st = {e.element: e.status for e in q.elements}
    assert st["Identify Pain"] == "confirmed"
    assert st["Economic Buyer"] == "partial"           # invented quote -> downgraded
    assert len(q.elements) == 8                         # missing elements filled as unknown
    assert any("Economic Buyer" in f for f in q.verification_flags)


def test_fit_score_and_priority():
    fits = [FitSignal(criterion_id="target_industry", assessment="met",
                      evidence_quote="24/7 lithium extraction sites in the Atacama Desert", note=""),
            FitSignal(criterion_id="warm_referral", assessment="met", evidence_quote="Referred by Anglo American", note=""),
            FitSignal(criterion_id="large_sites", assessment="met",
                      evidence_quote="drone-based inspection for 3 large-scale", note=""),
            FitSignal(criterion_id="budget_or_timeline_signal", assessment="partial",
                      evidence_quote="Internal budget conversation scheduled for Q3", note=""),
            FitSignal(criterion_id="continuous_ops", assessment="met", evidence_quote="totally made up quote here", note="")]
    q = score(_draft([], fits), EMAIL)
    # 20 + 10 + 15 + 5 = 50 points of 100; continuous_ops quote is fake -> 0
    assert q.fit_score == 50
    assert q.priority.startswith("P3")
    assert any("continuous_ops" in f for f in q.verification_flags)


# ---------------------------------------------------------------- sequence checker
def _email(step, ids, targets, body="Short body.", materials=()):
    return SequenceEmail(step=step, send_timing="d", trigger="t", goal="g", unknowns_targeted=targets,
                         finding_ids=list(ids), materials=list(materials), subject="s", body=body, why_this_step="w")


def test_sequence_checker():
    d = SequenceDraft(strategy="s", tone_notes="t", reply_branches=[], emails=[
        _email(1, ["F01"], ["Metrics"], materials=["https://v.com/cs"]),
        _email(2, ["F99"], ["Champion"], body="Hi [Name], " + "word " * 200)])
    issues, cov = check(d, {"F01"}, ["Metrics", "Champion", "Paper Process"], "https://v.com/cs")
    joined = " ".join(issues)
    assert "F99" in joined and "placeholder" in joined and "words" in joined
    assert "Paper Process" in joined
    assert cov["Metrics"] == [1]


def test_sequence_checker_passes():
    d = SequenceDraft(strategy="s", tone_notes="t", reply_branches=[], emails=[
        _email(1, ["F01"], ["Metrics", "Champion"], materials=["https://v.com/cs"])])
    issues, _ = check(d, {"F01"}, ["Metrics", "Champion"], "https://v.com/cs")
    assert issues == []


# ---------------------------------------------------------------- GTM rules
def test_gtm_rules():
    assert decide("high", [1], [])[0].startswith("Hybrid")
    assert decide("high", [], [])[0] == "Direct AE engagement"
    assert decide("medium", [1], [])[0].startswith("Partner-led with AE")
    assert decide("low", [], [1])[0] == "Partner-led"


# ---------------------------------------------------------------- site discovery
def test_sitemap_parsing_and_language_filter():
    idx = ('<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
           '<sitemap><loc>https://v.com/sitemap/case-studies</loc></sitemap></sitemapindex>')
    assert _locs(idx) == (["https://v.com/sitemap/case-studies"], [])
    us = ('<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
          '<url><loc>https://v.com/case-studies/a</loc></url><url><loc>https://v.com/es/case-studies/a</loc></url></urlset>')
    _, urls = _locs(us)
    assert [u for u in urls if is_default_language(u)] == ["https://v.com/case-studies/a"]
    assert section_of("https://v.com/blog/x-y") == "blog"
    assert is_default_language("https://v.com/tv/topics/customer-stories")  # 'tv' is a section, not a language
    assert not is_default_language("https://v.com/pt")                     # translated homepage
    assert not is_default_language("https://v.com/pt-br/blog/x")


# ---------------------------------------------------------------- SEC identity
class _FakeWeb:
    def __init__(self, tickers, fulltext):
        self.t, self.f = tickers, fulltext

    def get_json(self, url, headers=None):
        return self.t if "company_tickers" in url else self.f


class _Log:
    def info(self, m): pass


class _S:
    sec_user_agent = "t t@t.com"


def _ft(*rows):
    return {"hits": {"hits": [{"_source": {"ciks": [c], "display_names": [d]}} for c, d in rows]}}


def test_sec_ignores_companies_that_only_mention_the_lead():
    from agent.tools.sec import SEC
    potash = ("0000855931", "POTASH CORP OF SASKATCHEWAN INC  (CIK 0000855931)")
    sqm = ("0000909037", "CHEMICAL & MINING CO OF CHILE INC  (SQM)  (CIK 0000909037)")
    names = ["Sociedad Química y Minera de Chile", "SQM"]
    web = _FakeWeb({}, _ft(*([potash] * 40 + [sqm] * 3)))   # shareholder mentions SQM far more often
    r = SEC(web, _S(), _Log()).find_cik(names, ["SQM"])
    assert r["cik"] == "909037"
    web = _FakeWeb({"0": {"cik_str": 909037, "ticker": "SQM", "title": "CHEMICAL & MINING CO OF CHILE INC"}}, {})
    assert SEC(web, _S(), _Log()).find_cik(names, ["SQM"])["method"].startswith("ticker table")
    web = _FakeWeb({}, _ft(potash))
    assert SEC(web, _S(), _Log()).find_cik(names, ["SQM"]) is None
