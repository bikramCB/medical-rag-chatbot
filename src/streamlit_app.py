"""Local UI for the evidence API in src/app.py.

Start the API first, from the src directory:
    python app.py
Then:
    streamlit run streamlit_app.py
"""
import os
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
EXAMPLE_PDF = ROOT / "chatwithpdf" / "papers" / "Advances_nutritions.pdf"
PICO_FIELDS = (
    "population",
    "sample_size",
    "inclusion",
    "exclusion",
    "demographics",
    "intervention",
    "dosage",
    "route",
    "schedule",
    "comparator",
    "study_design",
    "trial_id",
    "funding",
    "conflicts_of_interest",
)


def api_base() -> str:
    return st.session_state.get("api_base", "http://127.0.0.1:5000").rstrip("/")


def post_json(path: str, payload: dict, timeout: int = 600):
    response = requests.post(f"{api_base()}{path}", json=payload, timeout=timeout)
    try:
        body = response.json()
    except ValueError:
        body = {"error": response.text[:500]}
    if response.status_code >= 400:
        raise RuntimeError(body.get("error") or response.text[:500])
    return body


def save_upload(uploaded, name: str) -> str:
    UPLOADS.mkdir(exist_ok=True)
    target = UPLOADS / name
    target.write_bytes(uploaded.getvalue())
    return str(target.resolve())


def field_value(card: dict, name: str) -> str:
    field = card.get(name) or {}
    if isinstance(field, dict):
        return field.get("value") or ""
    return ""


def show_card(card: dict):
    rows = []
    for name in PICO_FIELDS:
        value = field_value(card, name)
        field = card.get(name) or {}
        rows.append({
            "Field": name.replace("_", " "),
            "Value": value or "—",
            "Page": field.get("page") if isinstance(field, dict) else None,
            "Quote": (field.get("verbatim_quote") or "") if isinstance(field, dict) else "",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    outcomes = card.get("outcomes") or []
    if outcomes:
        st.markdown("**Outcomes**")
        outcome_rows = []
        for outcome in outcomes:
            stats = outcome.get("statistics") or {}
            outcome_rows.append({
                "Endpoint": outcome.get("endpoint"),
                "Role": outcome.get("role"),
                "Effect": outcome.get("effect"),
                "CI": outcome.get("ci"),
                "p": outcome.get("p_value"),
                "Crosses null": stats.get("crosses_null"),
                "Significant by CI": stats.get("significant_by_ci"),
                "Spin": stats.get("spin"),
                "Page": outcome.get("page"),
            })
        st.dataframe(outcome_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No grounded outcomes. A review with no printed hazard ratio or interval stays blank here.")


def show_appraisal(card: dict):
    appraisal = card.get("appraisal") or {}
    if not appraisal:
        st.info("Load the scorecard first. The checklist is part of that response.")
        return
    st.write(f"Checklist: {appraisal.get('checklist') or 'not set'}")
    st.caption("This is an automated checklist. It is not an official GRADE or Cochrane rating.")
    flags = appraisal.get("red_flags") or []
    if flags:
        for flag in flags:
            st.warning(f"{flag.get('flag')}: {flag.get('detail')}")
    else:
        st.success("No automatic red flags on this paper.")
    items = appraisal.get("items") or []
    if items:
        st.dataframe(
            [{"Item": item.get("item"), "Status": item.get("status"), "Note": item.get("note"), "Page": item.get("page")} for item in items],
            use_container_width=True,
            hide_index=True,
        )
    if appraisal.get("error"):
        st.error(appraisal["error"])


def show_live(card: dict):
    live = card.get("live_evidence") or {}
    if not live:
        st.info("Load the scorecard first. Retraction, trial, and citation lookups are part of that response.")
        return
    if live.get("skipped"):
        st.info("Live lookups are off. Set LIVE_EVIDENCE=true on the API and reload.")
        return
    retracted = live.get("retracted")
    if retracted:
        st.error("This DOI is marked retracted.")
    else:
        st.success("No retraction was found for the paper DOI.")
    crossref = live.get("crossref") or {}
    if crossref.get("title"):
        st.write(crossref["title"])
    alerts = live.get("cited_paper_alerts") or []
    if alerts:
        st.warning("A cited DOI has a retraction or correction.")
        st.json(alerts)
    else:
        st.caption("Cited DOIs that were checked have no retraction or correction notice.")
    trials = live.get("trials") or []
    if trials:
        for trial in trials:
            switch = trial.get("outcome_switch") or {}
            st.write(f"{trial.get('nct_id')}: {trial.get('status')}")
            if switch.get("flag"):
                st.warning(switch.get("reason"))
    else:
        st.caption("No NCT number was printed in the paper, so ClinicalTrials.gov was not called.")
    citations = live.get("citations") or {}
    newer = citations.get("newer_citing_papers") or []
    if citations.get("cited_by_count") is not None:
        st.write(f"Cited by {citations['cited_by_count']} papers. Newer citing papers are listed, not judged.")
    if newer:
        st.dataframe(newer, use_container_width=True, hide_index=True)
    for error in live.get("errors") or []:
        st.caption(error)


def show_statistics(card: dict, chat_result: dict | None):
    rows = []
    for outcome in (card or {}).get("outcomes") or []:
        stats = outcome.get("statistics") or {}
        if not stats:
            continue
        rows.append({"Source": outcome.get("endpoint") or "outcome", **{key: stats.get(key) for key in (
            "measure", "estimate", "ci_low", "ci_high", "crosses_null", "significant_by_ci", "significant_by_p", "ci_p_disagree", "spin"
        )}})
    for check in (chat_result or {}).get("quote_checks") or []:
        for stats in check.get("statistics") or []:
            rows.append({"Source": (check.get("claim") or "")[:80], **{key: stats.get(key) for key in (
                "measure", "estimate", "ci_low", "ci_high", "crosses_null", "significant_by_ci", "significant_by_p", "ci_p_disagree", "spin"
            )}})
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No printed interval has been audited yet. A review that never prints a confidence interval stays empty.")


def show_matrix(matrix: dict):
    rows = []
    for row in matrix.get("rows") or []:
        outcomes = row.get("primary_outcomes") or [{}]
        first = outcomes[0]
        stats = first.get("statistics") or {}
        rows.append({
            "Paper": row.get("source_pdf"),
            "Design": row.get("study_design"),
            "N": row.get("sample_size"),
            "Intervention": row.get("intervention"),
            "Comparator": row.get("comparator"),
            "Primary endpoint": first.get("endpoint"),
            "Effect": first.get("effect"),
            "CI": first.get("ci"),
            "Crosses null": stats.get("crosses_null"),
            "Page": first.get("page"),
        })
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    for item in matrix.get("contradictions") or []:
        st.warning(f"Disagreement: {item.get('endpoints')}")
    for item in matrix.get("consensus") or []:
        st.success(f"{item.get('count')} papers agree on {item.get('shared_term')} ({item.get('direction')}).")
    for error in matrix.get("errors") or []:
        st.error(f"{error.get('source_pdf')}: {error.get('error')}")


st.set_page_config(page_title="Evidence Intelligence", layout="wide")
st.title("Evidence Intelligence")
st.caption("This page calls the API in src/app.py. Scorecard, checklist, live lookups, and interval checks come back together. Compare papers uses /matrix.")

st.session_state.setdefault("api_base", "http://127.0.0.1:5000")
force = False

with st.sidebar:
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is not None:
    token = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("loaded_token") != token:
        pdf_url = save_upload(uploaded, uploaded.name)
        st.session_state["pdf_url"] = pdf_url
        st.session_state["loaded_token"] = token
        st.session_state["card"] = {}
        st.session_state["chat_result"] = {}
        st.session_state["history"] = []
        with st.spinner("Reading the paper. The first load parses, indexes, and checks outside sources."):
            try:
                st.session_state["card"] = post_json("/scorecard", {"pdf_url": pdf_url, "force": False})
                st.session_state["card_error"] = ""
            except Exception as exc:
                st.session_state["card_error"] = str(exc)

card = st.session_state.get("card") or {}
if st.session_state.get("card_error"):
    st.error(st.session_state["card_error"])

score_tab, appraisal_tab, live_tab, stats_tab, matrix_tab, chat_tab = st.tabs(
    ["Scorecard", "Appraisal", "Live evidence", "Statistics", "Compare papers", "Chat"]
)

with score_tab:
    if card:
        show_card(card)
    else:
        st.info("Load a PDF from the sidebar.")

with appraisal_tab:
    show_appraisal(card)

with live_tab:
    show_live(card)

with stats_tab:
    show_statistics(card, st.session_state.get("chat_result"))

with matrix_tab:
    st.write("One path or URL per line. Each paper becomes a row. A disagreement is shown only when both papers print related intervals that do not agree.")
    default_lines = st.session_state.get("pdf_url") or ""
    paper_list = st.text_area("Papers", value=default_lines, height=120)
    extra = st.file_uploader("Add more PDFs", type=["pdf"], accept_multiple_files=True, key="matrix_files")
    if st.button("Build matrix"):
        urls = [line.strip() for line in paper_list.splitlines() if line.strip()]
        for item in extra or []:
            urls.append(save_upload(item, item.name))
        if len(urls) < 1:
            st.warning("Add at least one paper.")
        else:
            with st.spinner("Building one scorecard per paper."):
                try:
                    st.session_state["matrix"] = post_json("/matrix", {"pdf_urls": urls, "force": force})
                    st.session_state["matrix_error"] = ""
                except Exception as exc:
                    st.session_state["matrix_error"] = str(exc)
    if st.session_state.get("matrix_error"):
        st.error(st.session_state["matrix_error"])
    if st.session_state.get("matrix"):
        show_matrix(st.session_state["matrix"])

with chat_tab:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    for turn in st.session_state["history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
    question = st.chat_input("Ask about the loaded paper")
    if question:
        pdf_url = st.session_state.get("pdf_url")
        st.session_state["history"].append({"role": "user", "content": question})
        if not pdf_url:
            st.session_state["history"].append({"role": "assistant", "content": "Load a PDF from the sidebar first."})
        else:
            prior = []
            messages = st.session_state["history"][:-1]
            for index in range(0, len(messages) - 1, 2):
                if messages[index]["role"] == "user" and messages[index + 1]["role"] == "assistant":
                    prior.append({"human": messages[index]["content"], "ai": messages[index + 1]["content"]})
            with st.spinner("Answering from the indexed paper."):
                try:
                    result = post_json("/chat", {"pdf_url": pdf_url, "query": question, "history": prior})
                    st.session_state["chat_result"] = result
                    st.session_state["history"].append({"role": "assistant", "content": result.get("answer") or ""})
                    if result.get("evidence_card"):
                        st.session_state["card"] = result["evidence_card"]
                except Exception as exc:
                    st.session_state["history"].append({"role": "assistant", "content": str(exc)})
        st.rerun()
    result = st.session_state.get("chat_result") or {}
    citations = result.get("citations") or []
    if citations:
        st.markdown("**Citations**")
        st.dataframe(
            [{"Chunk": item.get("chunk"), "Page": (item.get("metadata") or {}).get("page"), "Section": (item.get("metadata") or {}).get("section_category")} for item in citations],
            use_container_width=True,
            hide_index=True,
        )
        pdf_url = st.session_state.get("pdf_url")
        if pdf_url and st.button("Highlight these citations on the PDF"):
            try:
                painted = requests.post(
                    f"{api_base()}/highlight",
                    json={"pdf_url": pdf_url, "citations": citations},
                    timeout=120,
                )
                painted.raise_for_status()
                st.download_button("Download highlighted PDF", data=painted.content, file_name="highlighted.pdf", mime="application/pdf")
            except Exception as exc:
                st.error(str(exc))
