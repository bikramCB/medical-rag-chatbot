"""Stateless state/cache helpers and hierarchical map-reduce long-document synthesis."""
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from langchain.schema import Document

_STATE_DIR = Path(__file__).resolve().parent.parent / "state_store"
_LOCAL_LOCK = threading.Lock()
_LOCAL_CACHE: Dict[str, str] = {}
_REDIS_CLIENT = None
_REDIS_CHECKED = False


def _get_redis():
    global _REDIS_CLIENT, _REDIS_CHECKED
    if _REDIS_CHECKED:
        return _REDIS_CLIENT
    _REDIS_CHECKED = True
    url = (os.getenv("REDIS_URL") or "").strip()
    if not url:
        return None
    try:
        import redis
        client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        _REDIS_CLIENT = client
    except Exception as exc:
        print(f"WARNING: Redis unavailable ({exc}); using stateless disk/memory store.")
        _REDIS_CLIENT = None
    return _REDIS_CLIENT


def stateless_mode() -> bool:
    """Server requests stay stateless unless STATELESS_MODE is turned off for a single-user run."""
    raw = (os.getenv("STATELESS_MODE", "true") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _safe_key(namespace: str, key: str) -> str:
    raw = f"{namespace}:{key}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    clean_ns = re.sub(r"[^a-zA-Z0-9_-]", "_", namespace or "default")
    return f"medrag:{clean_ns}:{digest}"


def state_get(namespace: str, key: str) -> Optional[dict]:
    """Read JSON state by key from Redis or stateless disk fallback."""
    r_key = _safe_key(namespace, key)
    client = _get_redis()
    if client is not None:
        try:
            raw = client.get(r_key)
            return json.loads(raw) if raw else None
        except Exception:
            pass

    with _LOCAL_LOCK:
        if r_key in _LOCAL_CACHE:
            try:
                return json.loads(_LOCAL_CACHE[r_key])
            except Exception:
                pass
        path = _STATE_DIR / f"{r_key}.json"
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def state_set(namespace: str, key: str, value: dict, ttl_seconds: int = 86400) -> None:
    """Persist JSON state by key in Redis (with TTL) or stateless disk fallback."""
    r_key = _safe_key(namespace, key)
    payload = json.dumps(value, default=str)
    client = _get_redis()
    if client is not None:
        try:
            client.setex(r_key, max(60, int(ttl_seconds)), payload)
            return
        except Exception:
            pass

    with _LOCAL_LOCK:
        _LOCAL_CACHE[r_key] = payload
        try:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            (_STATE_DIR / f"{r_key}.json").write_text(payload, encoding="utf-8")
        except Exception:
            pass


def _group_chunks_for_map(
    documents: Sequence[Document],
    max_chars_per_group: int = 12000,
    max_groups: int = 12,
) -> List[tuple[str, str]]:
    """Group chunks by section/order into bounded text blocks for map-step summarization."""
    buckets: List[tuple[str, List[str], int]] = []
    current_label = "Overview"
    current_parts: List[str] = []
    current_chars = 0

    for doc in documents or []:
        meta = getattr(doc, "metadata", None) or {}
        sec = (meta.get("section_category") or "Document").strip()
        page = meta.get("page_number")
        chunk_index = meta.get("chunk_index")
        chunk_label = f"CHUNK_{chunk_index}" if chunk_index is not None else "CHUNK_?"
        header = f"[Chunk ID: {chunk_label} | Section: {sec} | Page: {page if page is not None else '?'}]"
        text = (getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue
        entry = f"{header}\n{text}"

        if current_parts and (
            (sec != current_label and current_chars >= max_chars_per_group // 2)
            or (current_chars + len(entry) > max_chars_per_group)
        ):
            buckets.append((current_label, current_parts, current_chars))
            current_label = sec
            current_parts = [entry]
            current_chars = len(entry)
        else:
            if not current_parts:
                current_label = sec
            current_parts.append(entry)
            current_chars += len(entry)

    if current_parts:
        buckets.append((current_label, current_parts, current_chars))

    # If there are too many buckets, merge tail buckets so total LLM map calls stay bounded
    if len(buckets) > max_groups:
        head = buckets[: max_groups - 1]
        tail_parts: List[str] = []
        tail_chars = 0
        for _lbl, parts, ch in buckets[max_groups - 1 :]:
            tail_parts.extend(parts)
            tail_chars += ch
        head.append(("Later Sections", tail_parts, tail_chars))
        buckets = head

    return [(label, "\n\n".join(parts)) for label, parts, _ch in buckets]


def hierarchical_summarize_documents(
    documents: Sequence[Document],
    llm_call: Callable[[str], str],
    focus_instruction: str = "Provide a structured clinical/technical summary preserving key numbers, endpoints, cohorts, and limitations.",
    direct_char_threshold: int = 14000,
    max_chars_per_group: int = 12000,
    max_groups: int = 10,
) -> str:
    """
    Summarize short documents directly and long documents via map-reduce across sections/groups.
    Avoids blind full-text concatenation and lost-in-the-middle degradation on long PDFs.
    """
    docs_list = list(documents or [])
    if not docs_list:
        return ""

    combined = "\n\n".join(
        (getattr(d, "page_content", "") or "").strip()
        for d in docs_list
        if (getattr(d, "page_content", "") or "").strip()
    )
    if not combined:
        return ""

    # Fast path for short/medium documents
    if len(combined) <= direct_char_threshold:
        prompt = (
            f"{focus_instruction}\n\n"
            f"Source excerpt:\n{combined}"
        )
        return (llm_call(prompt) or "").strip()

    # Map-reduce path for long documents (guidelines, CSRs, FDA reviews, multi-part bundles)
    groups = _group_chunks_for_map(
        docs_list,
        max_chars_per_group=max_chars_per_group,
        max_groups=max_groups,
    )
    mapped_notes: List[str] = []
    for idx, (sec_label, block_text) in enumerate(groups, start=1):
        map_prompt = (
            f"Extract concise, factual key points from part {idx}/{len(groups)} ({sec_label}). "
            f"Keep exact numbers, sample sizes, hazard ratios/confidence intervals, p-values, and limitations stated in the text. "
            f"When a point comes from a labeled chunk, keep its [CHUNK_N] tag. "
            f"Do not invent facts.\n\n{block_text[:max_chars_per_group]}"
        )
        note = (llm_call(map_prompt) or "").strip()
        if note:
            mapped_notes.append(f"### Part {idx}: {sec_label}\n{note}")

    if not mapped_notes:
        return (llm_call(f"{focus_instruction}\n\n{combined[:direct_char_threshold]}") or "").strip()

    reduce_prompt = (
        f"{focus_instruction}\n\n"
        f"Synthesize the section/part notes below into a single cohesive, non-repetitive summary. "
        f"Preserve exact numbers and explicitly note any conflicts or limitations across sections.\n\n"
        + "\n\n".join(mapped_notes)
    )
    return (llm_call(reduce_prompt) or "").strip()