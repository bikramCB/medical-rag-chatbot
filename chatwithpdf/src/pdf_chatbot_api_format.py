"""Chat API.

prompts.yml holds every prompt. This file chooses one.
weaviate_store.py stores and searches chunks. chunking.py builds them.
"""
import atexit
import concurrent.futures
import hashlib
import io
import json
import os
import re
import shutil
import string
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import List, Literal

import requests
import torch
import weaviate
from weaviate.classes.init import Auth
import yaml
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

for _env_path in (
    Path(__file__).resolve().parents[1] / ".env",
    Path(__file__).resolve().parents[1] / "chatwithpdf" / ".env",
    Path(__file__).resolve().parent / ".env",
):
    if _env_path.is_file():
        load_dotenv(_env_path)
load_dotenv()

from chunking import ChunkingMixin
from evidence import extract_evidence_card, load_cached_card, save_card
from stats_audit import audit_claim
from weaviate_store import EmbeddingGenerationError, WeaviateStoreMixin, rerank_documents


def vertex_chat_model():
    """Gemini on Vertex for every text step. Figures keep the separate vision model."""
    import httpx
    import google.auth
    from google.auth.transport.requests import Request as AuthRequest

    credentials, detected_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    project = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or detected_project
    # This project can call gemini-3.5-flash on the global endpoint. Flash-Lite is not enabled.
    location = os.getenv("VERTEX_CHAT_LOCATION") or os.getenv("VERTEX_VISION_LOCATION") or "global"
    model = os.getenv("VERTEX_CHAT_MODEL") or os.getenv("VERTEX_VISION_MODEL") or "gemini-3.5-flash"
    if not project:
        raise RuntimeError("Set VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT for Vertex Gemini.")
    if not model.startswith("google/"):
        model = f"google/{model}"
    host = "https://aiplatform.googleapis.com" if location == "global" else f"https://{location}-aiplatform.googleapis.com"
    base_url = f"{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"

    def _bearer(request):
        if not credentials.valid:
            credentials.refresh(AuthRequest())
        request.headers["Authorization"] = f"Bearer {credentials.token}"

    print(f"Chat model: Vertex {model} in {location}.")
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key="vertex",
        temperature=0.0,
        max_tokens=4096,
        extra_body={"google": {"thinking_config": {"thinking_budget": 0}}},
        http_client=httpx.Client(event_hooks={"request": [_bearer]}, timeout=120),
    )

_PROMPTS = yaml.safe_load(Path(__file__).with_name("prompts.yml").read_text(encoding="utf-8"))


def get_prompt(section: str, key: str):
    """Return one prompt block from prompts.yml."""
    return _PROMPTS[section][key]


class QueryRouter(BaseModel):
    """Routes a user query to the appropriate tool."""
    tool_name: Literal[
        "summary",
        "reference_extractor",
        "abstract_handler",
        "specific_qa",
        "key_findings_extractor",
        "methodology_summary",
        "objectives_extractor",
        "insights",
        "limitations_extractor",
        "title_generation",
        "description_generation",
    ] = Field(..., description="The name of the tool to use for the given query.")


class CitedClaim(BaseModel):
    """One numbered statement and the source sentence it must match."""
    claim: str = Field(..., description="A sentence from the draft answer that states a number.")
    verbatim_quote: str = Field(..., description="Exact text copied from one source chunk.")
    chunk_id: int = Field(..., description="The integer N in CHUNK_N.")


class ClaimList(BaseModel):
    claims: List[CitedClaim] = Field(default_factory=list)


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?(?:\s*%)?")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def verify_grounded_claims(claims, documents):
    """Accept a quote only when it appears in a chunk, and flag numbers the chunk does not contain."""
    normalized_docs = [_normalize_ws(doc.page_content) for doc in documents]
    compact_docs = [re.sub(r"\s+", "", doc.page_content or "") for doc in documents]
    results = []
    for claim in claims:
        if isinstance(claim, dict):
            claim_text = claim.get("claim") or ""
            quote = claim.get("verbatim_quote") or ""
            chunk_id = claim.get("chunk_id")
        else:
            claim_text = getattr(claim, "claim", "") or ""
            quote = getattr(claim, "verbatim_quote", "") or ""
            chunk_id = getattr(claim, "chunk_id", None)
        norm_quote = _normalize_ws(quote)
        matched_chunk = None
        if isinstance(chunk_id, int) and 0 <= chunk_id < len(normalized_docs) and norm_quote and norm_quote in normalized_docs[chunk_id]:
            matched_chunk = chunk_id
        elif norm_quote:
            for index, text in enumerate(normalized_docs):
                if norm_quote in text:
                    matched_chunk = index
                    break
        source = compact_docs[matched_chunk] if matched_chunk is not None else ""
        missing = []
        for number in _NUMBER_RE.findall(claim_text):
            compact = re.sub(r"\s+", "", number)
            if compact and compact not in source:
                missing.append(number)
        results.append({
            "claim": claim_text,
            "verbatim_quote": quote,
            "chunk_id": matched_chunk if matched_chunk is not None else chunk_id,
            "quote_found": matched_chunk is not None,
            "unsupported_numbers": missing,
            "statistics": audit_claim(claim_text),
        })
    return results


def apply_citations(answer, context_docs):
    """Turn CHUNK_n markers into page labels and build the UI citation list."""
    final_answer = answer
    final_citation_list = []
    if isinstance(answer, str) and "CHUNK" in answer.upper() and context_docs:

        def replace_chunks_with_pages(match):
            inner_content = match.group(1)
            page_citations = set()

            # Handle ranges like "CHUNK_0 - CHUNK_11"
            range_match = re.search(r'CHUNK[\s_](\d+)\s*-\s*CHUNK[\s_](\d+)', inner_content, re.IGNORECASE)
            if range_match:
                start_id = int(range_match.group(1))
                end_id = int(range_match.group(2))
                for chunk_id in range(start_id, end_id + 1):
                    if chunk_id < len(context_docs):
                        doc = context_docs[chunk_id]
                        page_num = doc.metadata.get("page_number", "N/A")
                        section = doc.metadata.get("section_category")
                        label = f"p. {page_num}"
                        if section and section != "Front matter":
                            label = f"{section}, {label}"
                        page_citations.add(label)
            else:
                # Handle comma-separated lists like "CHUNK_1, CHUNK_2"
                chunk_ids_str = re.findall(r'CHUNK[\s_](\d+)', inner_content, re.IGNORECASE)
                chunk_ids = [int(i) for i in chunk_ids_str]
                for chunk_id in chunk_ids:
                    if chunk_id < len(context_docs):
                        doc = context_docs[chunk_id]
                        page_num = doc.metadata.get("page_number", "N/A")
                        section = doc.metadata.get("section_category")
                        label = f"p. {page_num}"
                        if section and section != "Front matter":
                            label = f"{section}, {label}"
                        page_citations.add(label)
                    
            unique_pages = sorted(list(page_citations))
            return f" [{', '.join(unique_pages)}]"

        # Updated pattern to capture both lists and ranges
        citation_pattern = r'\[((?:CHUNK[\s_]\d+\s*-\s*CHUNK[\s_]\d+)|(?:CHUNK[\s_]\d+(?:, ?CHUNK[\s_]\d+)*))\]'
        final_answer = re.sub(citation_pattern, replace_chunks_with_pages, answer, flags=re.IGNORECASE).strip()

    # Part 2: Build the definitive citation list.
    if context_docs:
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get("source", "N/A")
            page = doc.metadata.get("page_number", "N/A")

            stored_index = doc.metadata.get("chunk_index")
            if isinstance(stored_index, int):
                chunk_id = f"doc_{stored_index}"
            else:
                chunk_id = "unknown"

            ordered_metadata = {
                "source": source,
                "chunk_id": chunk_id,
                "page": page,
                "section_category": doc.metadata.get("section_category", "Front matter"),
            }
            box = {}
            for key in ("x0", "y0", "x1", "y1"):
                value = doc.metadata.get(key)
                if isinstance(value, (int, float)):
                    box[key] = value
            if box:
                ordered_metadata["bbox"] = {"page": page, **box}

            # Same 0-based id the prompt showed the model (CHUNK_0, CHUNK_1, ...).
            final_citation_list.append({
                "chunk": f"CHUNK_{i}",
                "page_content": doc.page_content,
                "metadata": ordered_metadata
            })
    return final_answer, final_citation_list


def print_result(answer, citations):
    print("\n--- Answer ---")
    print(answer)
    if citations:
        print("\n--- Citations ---")
        print(json.dumps({"citation_snippets": citations}, indent=4))


def body_documents(documents):
    """Drop bibliography chunks. Keep the paper when nothing is labeled."""
    kept = [doc for doc in documents if doc.metadata.get("section_category") != "References"]
    return kept or list(documents)


def format_labeled_chunks(documents) -> str:
    """Show the model the same CHUNK_N ids that citation mapping uses."""
    parts = []
    for index, doc in enumerate(documents):
        section = doc.metadata.get("section_category", "Front matter")
        page = doc.metadata.get("page_number", "N/A")
        parts.append(
            f"### Chunk ID: CHUNK_{index}\nSection: {section}\nPage: {page}\n```text\n{doc.page_content}\n```"
        )
    return "\n\n".join(parts)


_CITED_CHUNK_RE = re.compile(r"CHUNK[\s_](\d+)", re.IGNORECASE)

# Whole words that are safe to expand even when the user types them in lowercase.
_UNAMBIGUOUS_TERMS = {
    "nsclc": "non-small cell lung cancer",
    "sclc": "small cell lung cancer",
    "hfref": "heart failure with reduced ejection fraction",
    "hfpef": "heart failure with preserved ejection fraction",
    "pembrolizumab": "Keytruda MK-3475",
    "keytruda": "pembrolizumab MK-3475",
    "nivolumab": "Opdivo BMS-936558",
    "opdivo": "nivolumab BMS-936558",
    "trastuzumab": "Herceptin",
    "herceptin": "trastuzumab",
    "bevacizumab": "Avastin",
    "avastin": "bevacizumab",
    "hba1c": "glycated hemoglobin",
}
# Short tokens that are also ordinary English. Expand only the uppercase form.
_UPPERCASE_ACRONYMS = {
    "MI": "myocardial infarction",
    "OS": "overall survival",
    "PFS": "progression-free survival",
    "DFS": "disease-free survival",
    "HR": "hazard ratio",
    "OR": "odds ratio",
    "RR": "risk ratio",
    "CI": "confidence interval",
    "AE": "adverse event",
    "SAE": "serious adverse event",
    "RCT": "randomized controlled trial",
    "ITT": "intention to treat",
    "NNT": "number needed to treat",
    "BMI": "body mass index",
}


def cited_chunk_indexes(answer: str, document_count: int):
    seen = []
    for raw in _CITED_CHUNK_RE.findall(answer or ""):
        index = int(raw)
        if 0 <= index < document_count and index not in seen:
            seen.append(index)
    return seen


def expand_known_medical_terms(question: str) -> str:
    """Add a fixed synonym list before the chat model expands the query."""
    additions = []
    seen = set()
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9\-]{1,}", question or ""):
        token = match.group(0)
        expansion = _UNAMBIGUOUS_TERMS.get(token.lower())
        if expansion is None and token.isupper():
            expansion = _UPPERCASE_ACRONYMS.get(token)
        if not expansion:
            continue
        key = expansion.lower()
        if key in seen or key in (question or "").lower():
            continue
        seen.add(key)
        additions.append(expansion)
    if not additions:
        return question
    return f"{question.strip()} {' '.join(additions)}"


class PaperChatBot(WeaviateStoreMixin, ChunkingMixin):
    _EMBED_BATCH_SIZE = 100
    _VALID_TOOLS = set(QueryRouter.model_fields["tool_name"].annotation.__args__)
    _TASKS = {
        "summary": "_answer_summary",
        "title_generation": "_answer_title",
        "description_generation": "_answer_description",
        "abstract_handler": "_answer_abstract",
        "reference_extractor": "_answer_references",
        "key_findings_extractor": "_answer_key_findings",
        "methodology_summary": "_answer_methodology",
        "objectives_extractor": "_answer_objectives",
        "insights": "_answer_insights",
        "limitations_extractor": "_answer_limitations",
        "specific_qa": "_answer_specific_qa",
    }

    def __init__(self, pdf_url: str):
        self.pdf_url = pdf_url
        self.normalized_pdf_url = self.pdf_url.split('?')[0]
        self.full_papers_content: str = ""
        self.all_documents: List[Document] = []
        self.evidence_card = None
        self.summarization_refine_chain = None
        self.pdf_file_bytes = None
        self.init_error = ""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"INFO: Using device: '{self.device}' for processing.")

        try:
            if os.path.isfile(self.pdf_url):
                print(f"Reading local PDF to generate Content Hash: {self.pdf_url}")
                with open(self.pdf_url, "rb") as handle:
                    pdf_bytes = handle.read()
            else:
                print("Downloading PDF to generate Content Hash...")
                response = requests.get(self.pdf_url, timeout=60)
                response.raise_for_status()
                pdf_bytes = response.content
            self.pdf_file_bytes = io.BytesIO(pdf_bytes)
            file_hash = hashlib.sha256(self.pdf_file_bytes.getvalue()).hexdigest()
            print(f"File Content Hash: {file_hash}")
            self.rag_index_name = file_hash
        except Exception as e:
            print(f"Warning: Could not download PDF or generate hash. Error: {e}")
            self.rag_index_name = "default_index"

        self.temp_dir = tempfile.mkdtemp()
        print(f"INFO: Created temporary working directory: {self.temp_dir}")

        def cleanup_temp():
            try:
                if os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)
            except Exception:
                pass
        atexit.register(cleanup_temp)

        self.llm = vertex_chat_model()

        self.rewrite_chain = self._create_rewrite_chain()
        self.last_interaction_time = time.time()
        self.inactivity_limit = 3600
        monitor_thread = threading.Thread(target=self._monitor_inactivity, daemon=True)
        monitor_thread.start()

        print("Attempting to connect to Weaviate...")
        http_host = os.getenv("WEAVIATE_HOST", "localhost").strip().strip('"')
        grpc_host = os.getenv("WEAVIATE_GRPC_HOST", http_host).strip().strip('"')
        http_port = int(os.getenv("WEAVIATE_http_port", 8080))
        grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
        api_key = os.getenv("WEAVIATE_API_KEY", "").strip()
        secure = os.getenv("WEAVIATE_HTTP_SECURE", "false").strip().lower() in ("1", "true", "yes")
        try:
            self.weaviate_client = weaviate.connect_to_custom(
                http_host=http_host,
                http_port=http_port,
                http_secure=secure,
                grpc_host=grpc_host,
                grpc_port=grpc_port,
                grpc_secure=secure,
                auth_credentials=Auth.api_key(api_key) if api_key else None,
            )

            def _close_weaviate_client():
                client = getattr(self, "weaviate_client", None)
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass

            atexit.register(_close_weaviate_client)

            if self.weaviate_client.is_ready():
                print(f"Successfully connected to Weaviate at http://{http_host}:{http_port}.")
                self.weaviate_class_name = self._get_weaviate_class_name()
                self._setup_weaviate_schema()
                if self._check_weaviate_for_data():
                    print(f"Found existing data for Content Hash {self.rag_index_name}. Loading...")
                    self._load_data_from_weaviate()
                    self.pdf_file_bytes = None
                else:
                    print("No existing data found for this content. processing document...")
                    self._load_and_process_documents()
            else:
                self.init_error = f"Weaviate at {http_host}:{http_port} did not become ready."
                print(f"CRITICAL: {self.init_error}")
        except Exception as e:
            self.init_error = f"Could not reach Weaviate at {http_host}:{http_port} (gRPC {grpc_host}:{grpc_port}). {e}"
            print(f"CRITICAL: {self.init_error}")

        client = getattr(self, "weaviate_client", None)
        ready = False
        if client is not None:
            try:
                ready = client.is_ready()
            except Exception:
                ready = False
        print(f"Weaviate is ready: {ready}")
        if self.all_documents and self.rag_index_name != "default_index":
            try:
                print("Building clinical scorecard...")
                self.get_evidence_card()
            except Exception as e:
                print(f"WARNING: Clinical scorecard was not built: {e}")

    def get_evidence_card(self, force: bool = False):
        """Return the cached PICO card, or extract it from the labeled chunks."""
        if (
            not force
            and isinstance(self.evidence_card, dict)
            and not self.evidence_card.get("error")
            and "appraisal" in self.evidence_card
        ):
            return self.evidence_card
        if not force:
            cached = load_cached_card(self.rag_index_name)
            if isinstance(cached, dict) and not cached.get("error"):
                if "appraisal" not in cached or "live_evidence" not in cached:
                    self._enrich_card(cached, force=False)
                    save_card(self.rag_index_name, cached)
                self.evidence_card = cached
                print(f"Loaded cached clinical scorecard for {self.rag_index_name}.")
                return cached
        if not self.all_documents:
            card = {
                "source_pdf": self.pdf_url,
                "content_hash": self.rag_index_name,
                "outcomes": [],
                "error": "No document chunks are loaded.",
            }
            self.evidence_card = card
            return card
        card = extract_evidence_card(
            self.llm,
            self.all_documents,
            get_prompt("evidence_card", "system"),
            self.pdf_url,
            self.rag_index_name,
        )
        if card.get("error"):
            print(card["error"])
        else:
            self._enrich_card(card, force=force)
            save_card(self.rag_index_name, card)
            print("Saved clinical scorecard.")
        self.evidence_card = card
        return card

    def _enrich_card(self, card: dict, force: bool = False):
        """Add interval checks, the checklist, and live registry lookups onto a card."""
        from intelligence import enrich_card

        try:
            enrich_card(
                card,
                self.all_documents,
                llm=self.llm,
                appraisal_prompt=get_prompt("appraisal", "system"),
                force=force,
            )
        except Exception as exc:
            print(f"WARNING: Evidence extras were not added: {exc}")
            card.setdefault("appraisal", {"items": [], "red_flags": [], "error": str(exc)})
            card.setdefault("live_evidence", {"errors": [str(exc)]})

    def _direct_route(self, query: str) -> str:
        """UI button text hits the YAML fast path. Free text uses the LLM router."""
        normalized_query = query.lower().strip().translate(str.maketrans("", "", string.punctuation))
        fast_path = get_prompt("router", "fast_path")
        if normalized_query in fast_path:
            return fast_path[normalized_query]
        return self._llm_route(query)

    def _llm_route(self, query: str) -> str:
        """Structured-output router for free-text questions that miss the fast path."""
        try:
            router = self.llm.with_structured_output(QueryRouter)
            result = router.invoke([
                SystemMessage(content=get_prompt("router", "instructions")),
                HumanMessage(content=query),
            ])
            if isinstance(result, QueryRouter):
                tool_name = result.tool_name
            elif isinstance(result, dict):
                tool_name = result.get("tool_name")
            else:
                tool_name = getattr(result, "tool_name", None)
            if tool_name in self._VALID_TOOLS:
                print(f"INFO: LLM router selected '{tool_name}'.")
                return tool_name
            print(f"WARNING: LLM router returned unexpected tool '{tool_name}'. Using specific_qa.")
        except Exception as e:
            print(f"WARNING: LLM router failed ({e}). Using specific_qa.")
        return "specific_qa"

    def _create_rewrite_chain(self):
        rewrite_prompt = PromptTemplate.from_template(get_prompt("rewrite", "template"))
        return rewrite_prompt | self.llm | StrOutputParser()

    def _expand_medical_query(self, standalone_question: str) -> str:
        """Add known medical synonyms, then ask the chat model for any others."""
        grounded = expand_known_medical_terms(standalone_question)
        try:
            prompt = get_prompt("expand", "template").format(query=grounded)
            expanded = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            expanded = expanded[:1000].strip()
            if expanded:
                print(f"Expanded search query: {expanded}")
                return expanded
        except Exception as e:
            print(f"Query expansion failed, using the known synonyms: {e}")
        return grounded

    def _retrieve(self, standalone_question: str, top_k: int = 8):
        """Hybrid search for 40 candidates, then a cross-encoder keeps the top chunks."""
        candidate_k = int(os.getenv("RETRIEVAL_CANDIDATES", "40"))
        expanded = self._expand_medical_query(standalone_question)
        docs = self._perform_hybrid_search(expanded, k=candidate_k)
        kept = rerank_documents(standalone_question, docs, top_n=top_k, device=self.device)
        print(f"Two-stage retrieval kept {len(kept)} of {len(docs)} candidates.")
        return kept

    def _summarize_table(self, table_text: str) -> str:
        """Two sentences used as the table's embedding. The chunk text stays the full table."""
        try:
            prompt = get_prompt("table_summary", "template").format(table_text=(table_text or "")[:6000])
            summary = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            if summary:
                print("Summarized a table for embedding.")
            return summary
        except Exception as e:
            print(f"Table summary failed, embedding the table text: {e}")
            return ""

    def _check_answer_quotes(self, answer: str, context_docs):
        """Ask for verbatim quotes, then drop any number the chunk does not contain."""
        if not answer or not context_docs:
            return []
        indexes = cited_chunk_indexes(answer, len(context_docs)) or list(range(len(context_docs)))
        if len(indexes) > 12:
            indexes = indexes[:12]
        parts = []
        for index in indexes:
            doc = context_docs[index]
            parts.append(f"### CHUNK_{index}\n{doc.page_content[:1500]}")
        try:
            checker = self.llm.with_structured_output(ClaimList)
            result = checker.invoke([
                SystemMessage(content=get_prompt("quote_check", "system").format(
                    answer=answer[:4000],
                    context_for_llm="\n\n".join(parts),
                )),
                HumanMessage(content="List the numbered claims and their verbatim quotes."),
            ])
            if isinstance(result, ClaimList):
                claims = result.claims
            elif isinstance(result, dict):
                claims = result.get("claims") or []
            else:
                claims = getattr(result, "claims", []) or []
            return verify_grounded_claims(claims, context_docs)
        except Exception as e:
            print(f"Quote check failed: {e}")
            return []

    def generate_from_all_chunks(self, task_description: str):
        """Generate text from the full paper, skipping retrieval."""
        if not self.all_documents:
            return {"answer": "No documents available to process.", "citations": []}
        full_context = "\n\n".join(doc.page_content for doc in self.all_documents)
        sys_msg = get_prompt("full_text", "system").format(
            task_description=task_description,
            full_context=full_context,
        )
        try:
            response = self.llm.invoke([SystemMessage(content=sys_msg)])
            return {"answer": response.content.strip(), "citations": []}
        except Exception as e:
            print(f"An error occurred during LLM invocation for full-text generation: {e}")
            return {"answer": f"An internal error occurred: {e}.", "citations": []}

    def _answer_summary(self, query, formatted_history):
        answer = "Could not generate an answer based on the provided document."
        context_docs = body_documents(self.all_documents)
        print("Executing: Full Document Summarization")
        summary_prompt = PromptTemplate.from_template(get_prompt("summary", "template"))
        summary_chain = summary_prompt | self.llm | StrOutputParser()
        answer = summary_chain.invoke({"pubtext": format_labeled_chunks(context_docs)})
        return answer, context_docs

    def _answer_title(self, query, formatted_history):
        answer = "Could not generate an answer based on the provided document."
        print("Executing: Title Generation")
        full_document_text = "\n\n".join(doc.page_content for doc in self.all_documents)
        title_prompt = PromptTemplate.from_template(get_prompt("title", "template"))
        title_chain = title_prompt | self.llm | StrOutputParser()
        answer = title_chain.invoke({
            "generation_prompt": get_prompt("title", "instructions").strip(),
            "document_text": full_document_text,
        })
        return answer, None

    def _answer_description(self, query, formatted_history):
        answer = "Could not generate an answer based on the provided document."
        print("Executing: Description Generation")
        full_document_text = "\n\n".join(doc.page_content for doc in self.all_documents)
        description_prompt = PromptTemplate.from_template(get_prompt("description", "template"))
        description_chain = description_prompt | self.llm | StrOutputParser()
        answer = description_chain.invoke({
            "generation_prompt": get_prompt("description", "instructions").strip(),
            "document_text": full_document_text,
        })
        return answer, None

    def _answer_abstract(self, query, formatted_history):
        answer = "Could not generate an answer based on the provided document."
        context_docs = []
        print("Executing: Abstract Handler")
        potential_abstracts = []

        for doc in self.all_documents:
            if doc.metadata.get("page_number", 99) > 3:
                continue
            match = re.search(r'\b(Abstract|Summary)\b|A\s*B\s*S\s*T\s*R\s*A\s*C\s*T', doc.page_content, re.IGNORECASE)
            if not match:
                continue
            extracted_text = doc.page_content[match.end():].strip()
            page_num = doc.metadata.get("page_number", "N/A")
            if extracted_text:
                potential_abstracts.append((extracted_text, page_num, doc))

        if not potential_abstracts:
            for doc in self.all_documents:
                if doc.metadata.get("page_number", 99) > 3:
                    continue
                keyword_match = re.search(r'\bKeywords\b', doc.page_content, re.IGNORECASE)
                if not keyword_match:
                    continue
                extracted_text = doc.page_content[:keyword_match.start()].strip()
                page_num = doc.metadata.get("page_number", "N/A")
                if extracted_text and len(extracted_text.split()) > 20:
                    potential_abstracts.append((extracted_text, page_num, doc))

        abstract_text = None
        page_num = "N/A"
        if potential_abstracts:
            potential_abstracts.sort(key=lambda item: (item[1], -len(item[0])))
            abstract_text, page_num, source_doc = potential_abstracts[0]
            context_docs = [source_doc]

        if abstract_text:
            sys_msg = get_prompt("abstract", "plain_language").format(abstract_text=abstract_text)
            response = self.llm.invoke([
                SystemMessage(content=sys_msg),
                HumanMessage(content=query),
            ])
            answer = f"{response.content} [Page: {page_num}]"
        else:
            print("No explicit abstract found. Attempting to identify and summarize semantically.")
            context_docs = [
                doc for doc in self.all_documents
                if doc.metadata.get("page_number", 99) <= 3
            ]
            if not context_docs:
                answer = "I could not find any content in the first three pages to search for an abstract."
            else:
                sys_msg = get_prompt("abstract", "semantic").format(
                    context_for_llm=format_labeled_chunks(context_docs)
                )
                response = self.llm.invoke([
                    SystemMessage(content=sys_msg),
                    HumanMessage(content="Please identify and summarize the abstract from the provided text."),
                ])
                answer = f"{response.content} [Page: {page_num}]"
        return answer, context_docs or None

    def _answer_references(self, query, formatted_history):
        answer = "Could not generate an answer based on the provided document."
        context_docs = None
        print("Executing: Reference Section Extraction")
        reference_content = []
        start_chunk_index = -1
        labeled_refs = [
            doc for doc in self.all_documents
            if doc.metadata.get("section_category") == "References"
        ]
        if labeled_refs:
            reference_content = [doc.page_content for doc in labeled_refs]
            context_docs = labeled_refs
        else:
            references_pattern = re.compile(r'^\s*(?:\d+\.?\s*)?References\b', re.IGNORECASE | re.MULTILINE)
            for i, doc in enumerate(self.all_documents):
                if not references_pattern.search(doc.page_content):
                    continue
                start_chunk_index = i
                print(f"Found 'References' section starting in chunk index {start_chunk_index} on page {doc.metadata.get('page_number')}.")
                try:
                    split_content = references_pattern.split(doc.page_content, 1)
                    if len(split_content) > 1:
                        reference_content.append(split_content[1].strip())
                except Exception as e:
                    print(f"Could not split content in the first reference chunk: {e}")
                reference_content.append(doc.page_content)
                break
            if start_chunk_index != -1:
                remaining_chunks = self.all_documents[start_chunk_index + 1:]
                for chunk in remaining_chunks:
                    reference_content.append(chunk.page_content)
                context_docs = self.all_documents[start_chunk_index:]
        if reference_content:
            messages_for_llm = [
                SystemMessage(content=get_prompt("references", "system")),
                HumanMessage(content="\n".join(reference_content)),
            ]
            response = self.llm.invoke(messages_for_llm)
            answer = response.content
        else:
            answer = "I could not find a 'References' section in the document."
        return answer, context_docs

    def _section_after_header(self, start_index, pattern):
        """Return the header chunk and the next non-reference chunk."""
        doc = self.all_documents[start_index]
        match = pattern.search(doc.page_content)
        if not match:
            return None, []
        used = [doc]
        if start_index + 1 < len(self.all_documents):
            next_doc = self.all_documents[start_index + 1]
            references_pattern = re.compile(r'^\s*(?:\d+\.?\s*)?References\b', re.IGNORECASE | re.MULTILINE)
            if next_doc.metadata.get("section_category") != "References" and not references_pattern.search(next_doc.page_content):
                used.append(next_doc)
        return match, used

    def _answer_key_findings(self, query, formatted_history):
        answer = "Could not generate an answer based on the provided document."
        context_docs = None
        print("Executing: Key Findings Extraction from Full Document")
        key_findings_pattern = re.compile(r'^\s*(?:\d+\.?\s*)?(?:Key Findings|Core findings|Highlights)\s*$', re.MULTILINE | re.IGNORECASE)
        conclusion_pattern = re.compile(r'^\s*(?:\d+\.?\s*)?(?:Conclusion(?:s)?\b.*)', re.MULTILINE | re.IGNORECASE)
        used_docs = []
        section_type = None

        for pattern, label in ((key_findings_pattern, "Key Findings"), (conclusion_pattern, "Conclusion")):
            for i, doc in enumerate(self.all_documents):
                if doc.metadata.get("section_category") == "References":
                    continue
                match, used = self._section_after_header(i, pattern)
                if not match:
                    continue
                section_type = label
                used_docs = used
                print(f"Found '{section_type}' header in chunk {i} on page {doc.metadata.get('page_number')}.")
                break
            if used_docs:
                break

        if used_docs:
            print(f"Step 2: Summarizing the identified '{section_type}' section.")
            sys_msg = get_prompt("key_findings", "section").format(
                section_type=section_type,
                found_section_content=format_labeled_chunks(used_docs),
            )
            response = self.llm.invoke([
                SystemMessage(content=sys_msg),
                HumanMessage(content=f"Summarize the {section_type} of the paper."),
            ])
            answer = response.content
            context_docs = used_docs
        else:
            print("Step 2: No explicit section header found. Falling back to comprehensive analysis of the full document.")
            context_docs = body_documents(self.all_documents)
            sys_msg = get_prompt("key_findings", "fallback").format(
                context_for_llm=format_labeled_chunks(context_docs)
            )
            response = self.llm.invoke([
                SystemMessage(content=sys_msg),
                HumanMessage(content="Generate a comprehensive summary of the key findings from the entire document."),
            ])
            answer = response.content
        return answer, context_docs

    def _answer_methodology(self, query, formatted_history):
        answer = "Could not generate an answer based on the provided document."
        print("Executing: Methodology Summary Extraction from Full Document")
        context_docs = body_documents(self.all_documents)
        methodology_prompt = PromptTemplate.from_template(get_prompt("methodology", "template"))
        methodology_chain = methodology_prompt | self.llm | StrOutputParser()
        answer = methodology_chain.invoke({"pubtext": format_labeled_chunks(context_docs)})
        return answer, context_docs

    def _answer_from_retrieval(self, query, formatted_history, prompt_section, prompt_key, human_message, extra_fields=None):
        standalone_question = self.rewrite_chain.invoke({"chat_history": formatted_history, "query": query})
        print(f"Step 0: Rewritten Standalone Question: \"{standalone_question}\"")
        print("Step 1: Two-stage retrieval (hybrid candidates, then cross-encoder)...")
        docs = self._retrieve(standalone_question)
        fields = {"context_for_llm": format_labeled_chunks(docs)}
        if extra_fields:
            fields.update(extra_fields)
        if prompt_section == "limitations":
            fields["standalone_question"] = standalone_question
        sys_msg = get_prompt(prompt_section, prompt_key).format(**fields)
        response = self.llm.invoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=human_message),
        ])
        return response.content, docs

    def _answer_objectives(self, query, formatted_history):
        print("Executing: Objectives Extraction")
        return self._answer_from_retrieval(
            query,
            formatted_history,
            "objectives",
            "system",
            "Extract the objectives/aims based on your instructions and the provided text.",
        )

    def _answer_insights(self, query, formatted_history):
        print("Executing: Insights Extraction")
        return self._answer_from_retrieval(
            query,
            formatted_history,
            "insights",
            "system",
            "Extract the key insights based on your instructions and the provided text snippets.",
        )

    def _answer_limitations(self, query, formatted_history):
        print("Executing: Limitations and Gaps Extraction (RAG)")
        return self._answer_from_retrieval(
            query,
            formatted_history,
            "limitations",
            "system",
            "Extract the limitations, gaps, and future research directions based on your instructions.",
        )

    def _answer_specific_qa(self, query, formatted_history):
        print("Executing: Hybrid Search and Q&A")
        standalone_question = self.rewrite_chain.invoke({"chat_history": formatted_history, "query": query})
        print(f"Step 0: Rewritten Standalone Question: \"{standalone_question}\"")
        print("Step 1: Two-stage retrieval (hybrid candidates, then cross-encoder)...")
        docs = self._retrieve(standalone_question)
        docs.sort(key=lambda doc: doc.metadata.get("page_number") or 0)
        sys_msg = get_prompt("specific_qa", "system").format(
            formatted_history=formatted_history,
            context_for_llm=format_labeled_chunks(docs),
        )
        response = self.llm.invoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=standalone_question),
        ])
        return response.content, docs

    def chat(self, query: str, history: list = None):
        """Pick a prompt from prompts.yml and answer with that task."""
        self.last_interaction_time = time.time()

        if not self.all_documents:
            print("[SYSTEM] User returned! Reloading text chunks from Database...")
            self._load_data_from_weaviate()
            if not self.all_documents:
                return {"answer": "Error reloading data from database.", "citations": []}

        if not self.all_documents:
            print("Error: No paper content loaded. Cannot answer questions.")
            return {"answer": "No documents available to process.", "citations": []}

        try:
            history_parts = []
            if history:
                for turn in history:
                    if turn.get("human"):
                        history_parts.append(f"user: {turn['human']}")
                    if turn.get("ai"):
                        history_parts.append(f"assistant: {turn['ai']}")
            formatted_history = "\n".join(history_parts)

            chosen_tool = self._direct_route(query)
            print(f"INTENT: Direct Router chose tool: {chosen_tool}")
            runner = getattr(self, self._TASKS.get(chosen_tool, "_answer_specific_qa"))
            answer, context_docs = runner(query, formatted_history)
            quote_checks = []
            if (
                context_docs
                and isinstance(answer, str)
                and chosen_tool not in {"reference_extractor", "title_generation", "description_generation"}
            ):
                quote_checks = self._check_answer_quotes(answer, context_docs)
            final_answer, final_citation_list = apply_citations(answer, context_docs)
            if context_docs and len(context_docs) > 12:
                cited = {f"CHUNK_{index}" for index in cited_chunk_indexes(answer, len(context_docs))}
                final_citation_list = [item for item in final_citation_list if item.get("chunk") in cited]
            print_result(final_answer, final_citation_list)
            payload = {
                "answer": final_answer,
                "citations": final_citation_list,
                "quote_checks": quote_checks,
            }
            if isinstance(self.evidence_card, dict):
                payload["evidence_card"] = self.evidence_card
            return payload
        except Exception as e:
            print(f"An error occurred during query processing: {e}")
            print(f"Exception Type: {type(e)}")
            traceback.print_exc()
            answer = f"An internal error occurred: {e}."
            print("\n--- Answer ---")
            print(answer)
            return {"answer": answer, "citations": []}


def generate_details_for_chatbot(chatbot, title_task, description_task):
    """Generate a title and a description for one paper in parallel."""
    print(f"\n--- Processing document: {chatbot.pdf_url} ---")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_title = executor.submit(chatbot.generate_from_all_chunks, title_task)
        future_description = executor.submit(chatbot.generate_from_all_chunks, description_task)
        title_result = future_title.result()
        description_result = future_description.result()
        info = {
            "source_pdf": chatbot.pdf_url,
            "title": title_result["answer"],
            "description": description_result["answer"],
        }
        print(f"\nGenerated Title:\n{info['title']}")
        print(f"\nGenerated Description:\n{info['description']}")
        return info


def main():
    pdf_urls = []
    try:
        print("\n--- World-Class Chat with Paper Tool ---")
        chatbots = []
        for url in pdf_urls:
            if url and url.strip():
                print(f"\nInitializing chatbot for: {url}")
                chatbot = PaperChatBot(pdf_url=url)
                if chatbot.all_documents:
                    chatbots.append(chatbot)
                else:
                    print(f"Could not initialize chatbot for {url}. Skipping.")

        if not chatbots:
            print("Exiting due to no documents being loaded or processed successfully.")
            return

        session_history = []
        print("\n--- Ready to chat with all loaded documents ---")
        print("Type 'generate details' to produce a title and description for each document.")
        while True:
            user_query = input("\nYour question: ")
            if user_query.lower() == "exit":
                print("Exiting chat. Goodbye!")
                break
            is_generate_details_command = "generate details" in user_query.lower()
            is_combined_query = (
                "generate a formal, concise title" in user_query.lower()
                and "generate a concise description" in user_query.lower()
            )
            if is_generate_details_command or is_combined_query:
                print("\n--- Answer ---")
                title_task = get_prompt("title", "cli_task")
                description_task = get_prompt("description", "cli_task")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {
                        executor.submit(generate_details_for_chatbot, chatbot, title_task, description_task): chatbot
                        for chatbot in chatbots
                    }
                    for future in concurrent.futures.as_completed(futures):
                        chatbot = futures[future]
                        try:
                            future.result()
                        except Exception as exc:
                            print(f"{chatbot.pdf_url} generated an exception: {exc}")
            elif chatbots:
                result = chatbots[0].chat(user_query, session_history)
                if result.get("answer"):
                    session_history.append({"human": user_query, "ai": result["answer"]})
    except Exception as e:
        print(f"Critical Error: {e}")


if __name__ == "__main__":
    main()
