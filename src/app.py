from flask import Flask, Response, request, jsonify
from pdf_chatbot_api_format import PaperChatBot, generate_details_for_chatbot, EmbeddingGenerationError
from highlight_pdf import paint_citation_boxes
from matrix import build_matrix
import os
import logging
import concurrent.futures
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

# This dictionary will cache chatbot instances based on the PDF URL.
# This is a simple in-memory cache. For a production system, you might
# want to use a more robust caching solution like Redis.
chatbot_cache = {}

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    Flask endpoint to interact with the PaperChatBot.
    Expects a JSON payload with 'pdf_url' and 'query'.
    'history' is optional.
    """
    data = request.get_json()

    pdf_url = data.get('pdf_url')
    query = data.get('query')
    history = data.get('history', [])  # history is optional

    if not pdf_url or not query:
        return jsonify({"error": "Missing 'pdf_url' or 'query' in request body"}), 400

    try:
        # Normalize the URL by removing query parameters to use as a stable cache key
        normalized_pdf_url = pdf_url.split('?')[0]
        
        # Check if a chatbot for this PDF URL is already in our cache
        if normalized_pdf_url in chatbot_cache:
            logger.info(f"Using cached chatbot for {pdf_url}")
            chatbot = chatbot_cache[normalized_pdf_url]
        else:
            logger.info(f"Creating new chatbot for {pdf_url}")
            # This might take some time as it downloads and processes the PDF
            chatbot = PaperChatBot(pdf_url=pdf_url)
            if not chatbot.all_documents:
                message = _startup_error(chatbot, pdf_url)
                logger.error(message)
                return jsonify({"error": message}), 500
            chatbot_cache[normalized_pdf_url] = chatbot
            logger.info(f"Successfully created and cached chatbot for {pdf_url}")

        result = chatbot.chat(query, history)
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
        normalized_pdf_url = pdf_url.split('?')[0]
        if normalized_pdf_url in chatbot_cache:
            logger.info(f"Using cached chatbot for scorecard {pdf_url}")
            chatbot = chatbot_cache[normalized_pdf_url]
        else:
            logger.info(f"Creating chatbot for scorecard {pdf_url}")
            chatbot = PaperChatBot(pdf_url=pdf_url)
            if not chatbot.all_documents:
                return jsonify({"error": _startup_error(chatbot, pdf_url)}), 500
            chatbot_cache[normalized_pdf_url] = chatbot

        card = chatbot.get_evidence_card(force=bool(data.get("force")))
        return jsonify(card)
    except EmbeddingGenerationError:
        return jsonify({"error": "We apologize, but we were unable to process your document at this time. This may be due to a temporary issue. Please try uploading the document again."}), 503
    except Exception as e:
        logger.exception(f"Scorecard failed for {pdf_url}")
        return jsonify({"error": str(e)}), 500


def _chatbot_for(pdf_url: str):
    normalized_pdf_url = pdf_url.split('?')[0]
    if normalized_pdf_url in chatbot_cache:
        return chatbot_cache[normalized_pdf_url]
    chatbot = PaperChatBot(pdf_url=pdf_url)
    if not chatbot.all_documents:
        raise RuntimeError(_startup_error(chatbot, pdf_url))
    chatbot_cache[normalized_pdf_url] = chatbot
    return chatbot


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
            cards.append(_chatbot_for(pdf_url).get_evidence_card(force=bool(data.get("force"))))
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

    # Define the tasks for title and description generation
    title_task = """Generate a formal, concise title for this document.
    Rules:
    Maximum 12 words.
    Academic and professional tone.
    Output only the title text.
    Do not include explanations, markdown symbols (##), or references to chunks."""

    description_task = """Generate a concise description for this document.
    Rules:
    2 to 4 sentences (maximum 80 words).
    Academic and professional tone.
    Output only the description text.
    Do not include headings, markdown symbols (##), or references to chunks.
    Do not explain how the description was formed."""

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


if __name__ == '__main__':
    # Host '0.0.0.0' makes it accessible from outside a container.
    app.run(host='0.0.0.0', port=5000, debug=False)

