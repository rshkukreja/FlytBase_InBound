# Walkthrough video outline (8–12 minutes)

| Time | Show | Say |
|---|---|---|
| 0:00–1:00 | README architecture diagram | The problem: an inbound BDR's first hour on a lead, systematised. One input (the email), six outputs. |
| 1:00–3:00 | `agent/pipeline.py`, `agent/config.py`, `agent/evidence.py` | Stages as modules with typed contracts. The LLM reads and judges; code verifies, scores and decides. All weights and rules live in one file. |
| 3:00–6:30 | Terminal: `python run.py input/lead_sqm.txt --no-cache` | Narrate the live log: EDGAR resolves SQM's filing name, queries are planned, findings are verified or rejected, the sitemap crawl finds the referral content outside `/case-studies`, partners are extracted, and the sequence checker forces revisions. |
| 6:30–9:00 | `report.html` top to bottom | At-a-glance box → MEDDPICC table (show a downgraded judgement if any) → research findings with quotes → rejected-claims appendix → case-study ranking table → GTM decision trace → handoff. |
| 9:00–10:00 | `runs/<id>/llm_calls/` | Every prompt and response is on disk: nothing is a black box. |
| 10:00–10:45 | `evidence.py` `verify_quote` + the test with a fabricated claim | **Design decision I'm proud of:** the grounding gate. Fabrication is the disqualifier, so it is enforced in code, not requested in a prompt. |
| 10:45–11:30 | README limitations | **Limitation:** sender verification and org charts are limited to the open web; next steps are a CRM/webhook trigger and enrichment APIs. |
| 11:30–12:00 | README "Why MEDDPICC" | Framework choice: enterprise multi-site mining deal, where procurement, the economic buyer and the champion decide the outcome. |

Tip: run once beforehand so pages are cached; for the recording, use `--no-cache` only if you want the full live research on screen.
