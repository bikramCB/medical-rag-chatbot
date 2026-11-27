"""Study bundle management and supplementary appendix routing helpers."""
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

_BUNDLE_DIR = Path(__file__).resolve().parent.parent / "bundles"
_CROSS_REF_RE = re.compile(
    r"\b(?:supplementary|supplement|appendix|appendices|table\s*s\d+|fig(?:ure)?\s*s\d+|protocol|sap|statistical analysis plan)\b",
    re.I,
)


def compute_study_id(files: Iterable[dict]) -> str:
    """Deterministic bundle ID from ordered (role, path/hash) entries."""
    normalized = []
    for item in files or []:
        if not isinstance(item, dict):
            continue
        role = (item.get("doc_role") or "main").strip().lower()
        ident = (item.get("content_hash") or item.get("pdf_url") or "").strip()
        if ident:
            normalized.append(f"{role}:{ident}")
    raw = "|".join(normalized) or "empty_bundle"
    return "study_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def bundle_path(study_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", study_id or "")
    return _BUNDLE_DIR / f"{safe}.json"


def save_study_bundle(bundle: dict) -> dict:
    _BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    sid = bundle.get("study_id") or compute_study_id(bundle.get("members") or [])
    bundle["study_id"] = sid
    bundle_path(sid).write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle


def find_bundle_for_pdf(pdf_url: str) -> dict | None:
    """Return the saved bundle whose main paper or a member file is this PDF."""
    target = (pdf_url or "").strip()
    if not target or not _BUNDLE_DIR.is_dir():
        return None
    for path in _BUNDLE_DIR.glob("*.json"):
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (bundle.get("main_pdf") or "").strip() == target:
            return bundle
        for member in bundle.get("members") or []:
            if (member.get("pdf_url") or "").strip() == target:
                return bundle
    return None


def load_study_bundle(study_id: str) -> dict | None:
    path = bundle_path(study_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def tag_documents_for_study(documents, study_id: str, doc_role: str = "main", source_label: str = ""):
    """Attach study_id, doc_role, and bundle_source to every document chunk."""
    role = (doc_role or "main").strip().lower()
    if role not in {"main", "supplement", "protocol"}:
        role = "main"
    for doc in documents or []:
        meta = getattr(doc, "metadata", None)
        if not isinstance(meta, dict):
            continue
        if study_id:
            meta["study_id"] = study_id
        meta["doc_role"] = role
        if source_label:
            meta["bundle_source"] = source_label
    return documents


def query_needs_supplement(query: str, context_docs=None) -> bool:
    """Return True when the user question or retrieved main-text chunks point to supplements/appendices."""
    if _CROSS_REF_RE.search(query or ""):
        return True
    for doc in context_docs or []:
        text = getattr(doc, "page_content", "") or ""
        if _CROSS_REF_RE.search(text):
            return True
    return False