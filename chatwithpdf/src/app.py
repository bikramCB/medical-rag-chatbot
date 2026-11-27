from flask import Flask, Response, request, jsonify
from pdf_chatbot_api_format import PaperChatBot, generate_details_for_chatbot, EmbeddingGenerationError, get_prompt
from highlight_pdf import paint_citation_boxes
from matrix import build_matrix
from pathlib import Path
from stateless_synthesis import state_get, state_set, stateless_mode
import os
import logging
import concurrent.futures
import threading
from contextlib import contextmanager
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def _startup_error(chatbot, pdf_url: str) -> str:
    detail = getattr(chatbot, "init_error", "") or ""
    if detail:
        return detail
    return f"Could not initialize chatbot for {pdf_url}."


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# Process-local Weaviate clients, keyed by PDF. Chunk text is not the source of truth:
# stateless mode drops it after each request and reloads it from Weaviate.
chatbot_cache = {}
_BOT_LOCKS = {}
_BOT_LOCKS_GUARD = threading.Lock()


def _lock_for(normalized_pdf_url: str):
    with _BOT_LOCKS_GUARD:
        lock = _BOT_LOCKS.get(normalized_pdf_url)
        if lock is None:
            lock = threading.Lock()
            _BOT_LOCKS[normalized_pdf_url] = lock
        return lock


def _document_record(chatbot, pdf_url: str) -> dict:
    from study_bundle import find_bundle_for_pdf
    normalized = pdf_url.split("?")[0]
    bundle = find_bundle_for_pdf(normalized) or find_bundle_for_pdf(pdf_url) or {}
    study_id = (bundle.get("study_id") or getattr(chatbot, "study_id", "") or "").strip()
    if study_id:
        chatbot.study_id = study_id
    return {
        "pdf_url": normalized,
        "doc_id": getattr(chatbot, "rag_index_name", "") or "",
        "study_id": study_id,
        "chunk_count": len(getattr(chatbot, "all_documents", []) or []),
    }


def _remember_document(chatbot, pdf_url: str) -> dict:
    record = _document_record(chatbot, pdf_url)
    state_set("documents", record["doc_id"] or record["pdf_url"], record)
    state_set("documents_by_url", record["pdf_url"], record)
    return record


def _load_bot(pdf_url: str, force: bool = False):
    """Open the PDF's Weaviate client. Identity comes from the request and the shared store."""
    normalized = pdf_url.split("?")[0]
    known = state_get("documents_by_url", normalized)
    if known:
        logger.info("Resolved doc_id=%s study_id=%s", known.get("doc_id"), known.get("study_id"))
    chatbot = None if force else chatbot_cache.get(normalized)
    if chatbot is None:
        logger.info("Creating chatbot for %s", pdf_url)
        chatbot = PaperChatBot(pdf_url=pdf_url)
        if not chatbot.all_documents:
            raise RuntimeError(_startup_error(chatbot, pdf_url))
        chatbot_cache[normalized] = chatbot
    elif not chatbot.all_documents:
        logger.info("Reloading chunks for %s from Weaviate", pdf_url)
        chatbot._load_data_from_weaviate()
        if not chatbot.all_documents:
            raise RuntimeError(_startup_error(chatbot, pdf_url))
    record = _remember_document(chatbot, normalized)
    return chatbot, list(chatbot.all_documents), record


@contextmanager
def _document_bot(pdf_url: str, force: bool = False):
    normalized = pdf_url.split("?")[0]
    lock = _lock_for(normalized)
    lock.acquire()
    chatbot = None
    try:
        chatbot, docs, record = _load_bot(pdf_url, force=force)
        yield chatbot, docs, record
    finally:
        if chatbot is not None and stateless_mode():
            chatbot.release_loaded_text()
        lock.release()


def _session_history(pdf_url: str, session_id: str, request_history) -> list:
    if request_history:
        return list(request_history)
    if not session_id:
        return []
    saved = state_get("sessions", session_id) or state_get("sessions", pdf_url.split("?")[0])
    history = (saved or {}).get("history") or []
    return history if isinstance(history, list) else []


def _store_session(pdf_url: str, session_id: str, record: dict, history: list, query: str, answer: str):
    updated = list(history) + [{"human": query, "ai": answer}]
    payload = {
        "pdf_url": pdf_url.split("?")[0],
        "doc_id": record.get("doc_id") or "",
        "study_id": record.get("study_id") or "",
        "history": updated[-20:],
    }
    state_set("sessions", session_id or payload["pdf_url"], payload)


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    Flask endpoint to interact with the PaperChatBot.
    Expects a JSON payload with 'pdf_url' and 'query'.
    'history' and 'session_id' are optional.
    """
    data = request.get_json()

    pdf_url = data.get('pdf_url')
    query = data.get('query')
    session_id = (data.get("session_id") or "").strip()
    request_history = data.get('history') or []

    if not pdf_url or not query:
        return jsonify({"error": "Missing 'pdf_url' or 'query' in request body"}), 400

    try:
        with _document_bot(pdf_url) as (chatbot, _docs, record):
            history = _session_history(pdf_url, session_id, request_history)
            result = chatbot.chat(query, history)
            _store_session(pdf_url, session_id, record, history, query, result.get("answer") or "")
            result["doc_id"] = record.get("doc_id") or ""
            result["study_id"] = record.get("study_id") or ""
        return jsonify(result)
    
    except EmbeddingGenerationError as e:
        logger.error(f"Failed to create chatbot for {pdf_url} due to embedding generation error: {e}")
        return jsonify({"error": "We apologize, but we were unable to process your document at this time. This may be due to a temporary issue. Please try uploading the document again."}), 503

    except Exception as e:
        logger.exception(f"An error occurred while processing request for {pdf_url}")
        return jsonify({"error": str(e)}), 500

def _read_pdf_bytes(pdf_url: str) -> bytes:
    if os.path.isfile(pdf_url):
        with open(pdf_url, "rb") as handle:
            return handle.read()
    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()
    return response.content


@app.route('/highlight', methods=['POST'])
def highlight_endpoint():
    """Return the source PDF with a highlight annotation on each citation box."""
    data = request.get_json(silent=True) or {}
    pdf_url = data.get("pdf_url")
    citations = data.get("citations") or []
    if not pdf_url:
        return jsonify({"error": "Missing 'pdf_url' in request body"}), 400
    try:
        painted = paint_citation_boxes(_read_pdf_bytes(pdf_url), citations)
        return Response(
            painted,
            mimetype="application/pdf",
            headers={"Content-Disposition": "inline; filename=highlighted.pdf"},
        )
    except Exception as e:
        logger.exception(f"Highlight failed for {pdf_url}")
        return jsonify({"error": str(e)}), 500


@app.route('/scorecard', methods=['POST'])
def scorecard_endpoint():
    """Return the clinical PICO card for one PDF. The card is built once and cached by file hash."""
    data = request.get_json(silent=True) or {}
    pdf_url = data.get('pdf_url')
    if not pdf_url:
        return jsonify({"error": "Missing 'pdf_url' in request body"}), 400

    try:
        with _document_bot(pdf_url) as (chatbot, _docs, record):
            card = chatbot.get_evidence_card(force=bool(data.get("force")))
            card["doc_id"] = record.get("doc_id") or ""
            if record.get("study_id"):
                card["study_id"] = record["study_id"]
        return jsonify(card)
    except EmbeddingGenerationError:
        return jsonify({"error": "We apologize, but we were unable to process your document at this time. This may be due to a temporary issue. Please try uploading the document again."}), 503
    except Exception as e:
        logger.exception(f"Scorecard failed for {pdf_url}")
        return jsonify({"error": str(e)}), 500


def _get_or_build_bot(pdf_url: str, force: bool = False):
    chatbot, docs, _record = _load_bot(pdf_url, force=force)
    return chatbot, docs


@app.route('/matrix', methods=['POST'])
def matrix_endpoint():
    """Compare scorecards for several PDFs. Cells keep the page and box of the source outcome."""
    data = request.get_json(silent=True) or {}
    pdf_urls = data.get("pdf_urls")
    if not pdf_urls or not isinstance(pdf_urls, list):
        return jsonify({"error": "Missing 'pdf_urls' list in request body"}), 400
    cards = []
    errors = []
    for pdf_url in pdf_urls:
        if not pdf_url or not str(pdf_url).strip():
            continue
        try:
            with _document_bot(pdf_url) as (chatbot, _docs, _record):
                cards.append(chatbot.get_evidence_card(force=bool(data.get("force"))))
        except Exception as exc:
            logger.exception(f"Matrix skipped {pdf_url}")
            errors.append({"source_pdf": pdf_url, "error": str(exc)})
    matrix = build_matrix(cards)
    matrix["errors"] = errors
    return jsonify(matrix)


@app.route('/batch_generate_details', methods=['POST'])
def batch_generate_details_endpoint():
    """
    Flask endpoint to generate titles and descriptions for multiple PDFs in parallel.
    Expects a JSON payload with 'pdf_urls'.
    """
    data = request.get_json()
    pdf_urls = data.get('pdf_urls')

    if not pdf_urls or not isinstance(pdf_urls, list):
        return jsonify({"error": "Missing 'pdf_urls' list in request body"}), 400

    title_task = get_prompt("title", "instructions")
    description_task = get_prompt("description", "instructions")

    all_generated_info = []
    
    # --- Step 1: Initialize all chatbots ---
    # This step involves downloading and processing PDFs, which can be time-consuming.
    # A ThreadPoolExecutor is used to do this in parallel for multiple URLs.
    chatbots_to_process = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {executor.submit(PaperChatBot, pdf_url=url): url for url in pdf_urls if url and url.strip()}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                chatbot = future.result()
                if chatbot.all_documents:
                    chatbots_to_process.append(chatbot)
                else:
                    logger.warning(f"Could not initialize chatbot for {url}. Skipping.")
                    all_generated_info.append({"source_pdf": url, "error": "Could not initialize chatbot or document is empty."})
            except Exception as exc:
                logger.error(f"Error initializing chatbot for {url}: {exc}")
                all_generated_info.append({"source_pdf": url, "error": str(exc)})

    # --- Step 2: Generate titles and descriptions in parallel ---
    # This step uses the successfully initialized chatbots to generate content.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_chatbot = {
            executor.submit(generate_details_for_chatbot, chatbot, title_task, description_task): chatbot 
            for chatbot in chatbots_to_process
        }
        for future in concurrent.futures.as_completed(future_to_chatbot):
            chatbot_instance = future_to_chatbot[future]
            try:
                info = future.result()
                all_generated_info.append(info)
            except Exception as exc:
                logger.error(f'Error generating details for {chatbot_instance.pdf_url}: {exc}')
                all_generated_info.append({"source_pdf": chatbot_instance.pdf_url, "error": str(exc)})
    
    return jsonify(all_generated_info)


from study_bundle import compute_study_id, save_study_bundle, tag_documents_for_study


@app.post("/bundle")
def build_study_bundle():
    data = request.get_json(silent=True) or {}
    main_pdf = (data.get("main_pdf") or "").strip()
    supplement_pdfs = [p.strip() for p in (data.get("supplement_pdfs") or []) if isinstance(p, str) and p.strip()]
    protocol_pdfs = [p.strip() for p in (data.get("protocol_pdfs") or []) if isinstance(p, str) and p.strip()]
    force = bool(data.get("force", False))

    if not main_pdf:
        return jsonify({"error": "main_pdf is required"}), 400

    members = [{"doc_role": "main", "pdf_url": main_pdf}]
    members.extend({"doc_role": "supplement", "pdf_url": p} for p in supplement_pdfs)
    members.extend({"doc_role": "protocol", "pdf_url": p} for p in protocol_pdfs)

    study_id = compute_study_id(members)
    indexed_files = []

    for entry in members:
        role = entry["doc_role"]
        pdf_path = entry["pdf_url"]
        with _document_bot(pdf_path, force=force) as (bot, docs, _record):
            tag_documents_for_study(docs, study_id=study_id, doc_role=role, source_label=Path(pdf_path).name)
            bot.study_id = study_id
            bot.doc_role = role
            if hasattr(bot, "_cached_study_bundle"):
                del bot._cached_study_bundle
            tagged = bot.persist_study_tags(docs)
            if docs and tagged == 0:
                raise RuntimeError(
                    f"Study tags were not saved for {pdf_path}. Supplement search would miss this file."
                )
        indexed_files.append({
            "doc_role": role,
            "pdf_url": pdf_path,
            "chunk_count": len(docs or []),
            "tagged_chunks": tagged,
        })

    with _document_bot(main_pdf, force=force) as (main_bot, _main_docs, _record):
        main_bot.study_id = study_id
        if hasattr(main_bot, "_cached_study_bundle"):
            del main_bot._cached_study_bundle
        card = main_bot.get_evidence_card(force=force)
    card["study_id"] = study_id
    state_set("studies", study_id, {
        "study_id": study_id,
        "main_pdf": main_pdf,
        "doc_id": card.get("content_hash") or card.get("doc_id") or "",
        "members": indexed_files,
    })

    bundle = save_study_bundle({
        "study_id": study_id,
        "main_pdf": main_pdf,
        "supplement_pdfs": supplement_pdfs,
        "protocol_pdfs": protocol_pdfs,
        "members": indexed_files,
    })

    return jsonify({
        "study_id": study_id,
        "bundle": bundle,
        "evidence_card": card,
    })


if __name__ == '__main__':
    # Host '0.0.0.0' makes it accessible from outside a container.
    app.run(host='0.0.0.0', port=5000, debug=False)

