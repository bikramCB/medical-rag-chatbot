import gc
import os
import random
import re
import time
from typing import List
import requests

from langchain.schema import Document
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.query import Filter, HybridFusion

_MESH_CACHE: dict[str, list[str]] = {}


def fetch_mesh_synonyms(term: str, max_terms: int = 4, timeout: float = 3.0) -> list[str]:
    """Query NCBI E-utilities MeSH database with in-memory caching and safe fallback."""
    cleaned = re.sub(r"\s+", " ", (term or "").strip()).lower()
    if len(cleaned) < 3:
        return []
    if cleaned in _MESH_CACHE:
        return _MESH_CACHE[cleaned][:max_terms]

    synonyms: list[str] = []
    try:
        search_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "mesh", "term": cleaned, "retmode": "json", "retmax": 3},
            timeout=timeout,
        )
        search_resp.raise_for_status()
        id_list = ((search_resp.json().get("esearchresult") or {}).get("idlist")) or []
        if id_list:
            sum_resp = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "mesh", "id": ",".join(id_list[:3]), "retmode": "json"},
                timeout=timeout,
            )
            sum_resp.raise_for_status()
            result_map = sum_resp.json().get("result") or {}
            for uid in id_list[:3]:
                entry = result_map.get(str(uid)) or {}
                heading = ((entry.get("ds_meshterms") or [None])[0] or "").strip()
                if not heading:
                    heading = (entry.get("title") or "").strip()
                if heading and heading.lower() != cleaned and heading.lower() not in {s.lower() for s in synonyms}:
                    synonyms.append(heading)
    except Exception:
        synonyms = []

    _MESH_CACHE[cleaned] = synonyms
    return synonyms[:max_terms]
    
class EmbeddingGenerationError(Exception):
    """Raised when one or more chunk embeddings cannot be generated."""


def _should_retry_embedding(exc: Exception) -> bool:
    """Retry rate limits and server errors. A missing project or a bad response will not succeed later."""
    message = str(exc)
    if "Set VERTEX_PROJECT" in message or "did not contain a vector" in message or "vectors for" in message:
        return False
    if isinstance(exc, EmbeddingGenerationError):
        return any(code in message for code in ("429", "500", "502", "503", "504"))
    return True


def embedding_model_name() -> str:
    """Vertex publisher model id. text-embedding-004 matches the vectors already in use."""
    return os.getenv("VERTEX_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL_NAME") or "text-embedding-004"
# PubMed-trained cross-encoder. Smaller than bge-reranker-v2-m3, so a local rerank is faster.
# This is not Vertex's hosted ranker. Set RERANKER_MODEL_NAME to switch models.
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "ncbi/MedCPT-Cross-Encoder")
BBOX_KEYS = ("x0", "y0", "x1", "y1")

_reranker = None
_reranker_failed = False


def text_to_embed(doc: Document) -> str:
    """Embed table summaries or full text enriched with normalized biomedical entity aliases."""
    summary = doc.metadata.get("embed_text")
    base_text = summary.strip() if (isinstance(summary, str) and summary.strip()) else (doc.page_content or "")

    aliases = doc.metadata.get("entity_aliases")
    if not isinstance(aliases, list) or not aliases:
        try:
            from biomedical_nern import normalize_biomedical_text
            aliases = normalize_biomedical_text(doc.page_content or "", max_extra_terms=6).get("expansion_terms") or []
        except Exception:
            aliases = []

    if aliases:
        alias_line = "Related biomedical terms: " + ", ".join(str(a) for a in aliases if a)
        return f"{base_text}\n{alias_line}".strip()
    return base_text


def _text_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split("|") if part.strip()]
    return []


def bbox_properties(metadata: dict) -> dict:
    values = {}
    for key in BBOX_KEYS:
        raw = metadata.get(key)
        if isinstance(raw, (int, float)):
            values[key] = float(raw)
    return values


def metadata_from_properties(properties: dict) -> dict:
    raw_index = properties.get("chunk_index")
    metadata = {
        "source": properties.get("source"),
        "page_number": properties.get("page_number"),
        "element_type": properties.get("element_type"),
        "token_size": properties.get("token_size"),
        "chunk_index": int(raw_index) if raw_index is not None else None,
        "section_category": properties.get("section_category") or "Front matter",
        "study_id": properties.get("study_id") or "",
        "doc_role": properties.get("doc_role") or "main",
        "concept_ids": _text_list(properties.get("concept_ids")),
        "entity_aliases": _text_list(properties.get("entity_aliases")),
    }
    for key in BBOX_KEYS:
        raw = properties.get(key)
        if isinstance(raw, (int, float)):
            metadata[key] = float(raw)
    return metadata


def order_by_scores(documents: List[Document], scores: List[float], top_n: int) -> List[Document]:
    ranked = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def passages_for_rerank(doc: Document, window: int = 1500, stride: int = 800, max_windows: int = 4) -> List[str]:
    """Split a long table into overlapping windows so a later row can still win the score.

    A table summary, when one was stored on the chunk, is scored as its own window.
    """
    passages: List[str] = []
    summary = doc.metadata.get("embed_text")
    if isinstance(summary, str) and summary.strip():
        passages.append(summary.strip())
    text = doc.page_content or ""
    if not text.strip():
        return passages or [""]
    if len(text) <= window:
        passages.append(text)
        return passages
    starts = list(range(0, len(text), stride))
    if len(starts) > max_windows:
        last = len(starts) - 1
        starts = [starts[round(i * last / (max_windows - 1))] for i in range(max_windows)]
    for start in starts:
        passages.append(text[start:start + window])
    return passages


def rerank_documents(query: str, documents: List[Document], top_n: int = 8, device: str = "cpu") -> List[Document]:
    """Score hybrid candidates with a cross-encoder and keep the top ones.

    A failed model download leaves the hybrid order in place so chat still answers.
    """
    global _reranker, _reranker_failed
    if not documents:
        return []
    if top_n >= len(documents):
        top_n = len(documents)
    enabled = os.getenv("RERANKER_ENABLED", "true").strip().lower()
    if enabled in ("0", "false", "no") or _reranker_failed:
        return documents[:top_n]

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if _reranker is None:
            print(f"Loading cross-encoder {RERANKER_MODEL_NAME} on {device}...")
            tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
            model.to(device)
            model.eval()
            _reranker = (tokenizer, model)

        tokenizer, model = _reranker
        pairs = []
        for doc_index, doc in enumerate(documents):
            for passage in passages_for_rerank(doc):
                pairs.append((doc_index, passage))
        scores = [float("-inf")] * len(documents)
        with torch.no_grad():
            for start in range(0, len(pairs), 8):
                batch = pairs[start:start + 8]
                inputs = tokenizer(
                    [query] * len(batch),
                    [passage for _, passage in batch],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                logits = model(**inputs).logits.view(-1).float().cpu().tolist()
                for (doc_index, _), score in zip(batch, logits):
                    if score > scores[doc_index]:
                        scores[doc_index] = score
        print(f"Cross-encoder scored {len(documents)} candidates across {len(pairs)} passages.")
        return order_by_scores(documents, scores, top_n)
    except Exception as e:
        _reranker_failed = True
        print(f"WARNING: Cross-encoder rerank failed ({e}). Using hybrid order.")
        return documents[:top_n]


class WeaviateStoreMixin:
    def _get_weaviate_class_name(self):
        # Weaviate class names must start with a capital letter and contain only alphanumeric characters.
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', self.rag_index_name)
        if not sanitized or not sanitized[0].isalpha():
            sanitized = "Pdf" + sanitized
        return sanitized[0].upper() + sanitized[1:]

    def _monitor_inactivity(self):
        """Background thread to kill RAM if user is idle."""
        while True:
            time.sleep(60)  # Check every 1 minute
            elapsed = time.time() - self.last_interaction_time
            
            # If idle for > 1 hour AND we are currently holding data
            if elapsed > self.inactivity_limit and self.all_documents:
                print(f"\n[SYSTEM] Inactivity detected ({elapsed/60:.0f} mins). Clearing RAM to save resources...")
                
                # 1. Kill the RAM
                self.all_documents = []
                self.pdf_file_bytes = None
                
                # 2. Force Garbage Collection
                gc.collect()
                
                print("[SYSTEM] RAM cleared. Data will auto-reload on next query.")
                # We log memory usage to prove it worked

    def _setup_weaviate_schema(self):
        if not self.weaviate_client or not self.weaviate_client.is_ready():
            print("ERROR: Weaviate client is not ready in _setup_weaviate_schema.")
            return

        try:
            stored_model_name = ""
            collection_exists = self.weaviate_client.collections.exists(self.weaviate_class_name)

            if collection_exists:
                try:
                    config = self.weaviate_client.collections.get(self.weaviate_class_name).config.get()
                    description = config.description
                    if description and "Embedding model:" in description:
                        stored_model_name = description.split("Embedding model:")[1].strip()
                except Exception as e:
                    print(f"Warning: Could not retrieve model info from existing collection's description: {e}")

            # Re-index if collection exists but model is different
            if collection_exists and stored_model_name != embedding_model_name():
                print(f"Collection '{self.weaviate_class_name}' exists but was created with a different embedding model ('{stored_model_name}').")
                print("Deleting collection to re-index with the current model.")
                self.weaviate_client.collections.delete(self.weaviate_class_name)
                collection_exists = False

            if not collection_exists:
                print(f"Creating collection '{self.weaviate_class_name}'...")
                self.weaviate_client.collections.create(
                    name=self.weaviate_class_name,
                    description=f"Embedding model: {embedding_model_name()}",
                    properties=[
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="source", data_type=DataType.TEXT),
                        Property(name="page_number", data_type=DataType.INT),
                        Property(name="element_type", data_type=DataType.TEXT),
                        Property(name="token_size", data_type=DataType.INT),
                        Property(name="chunk_index", data_type=DataType.INT),
                        Property(name="section_category", data_type=DataType.TEXT),
                        Property(name="study_id", data_type=DataType.TEXT),
                        Property(name="doc_role", data_type=DataType.TEXT),
                        Property(name="concept_ids", data_type=DataType.TEXT_ARRAY),
                        Property(name="entity_aliases", data_type=DataType.TEXT_ARRAY),
                        Property(name="x0", data_type=DataType.NUMBER),
                        Property(name="y0", data_type=DataType.NUMBER),
                        Property(name="x1", data_type=DataType.NUMBER),
                        Property(name="y1", data_type=DataType.NUMBER),
                    ],
                    vectorizer_config=Configure.Vectorizer.none(),
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE
                    )
                )
                print(f"Successfully created collection '{self.weaviate_class_name}' with metadata for model '{embedding_model_name()}'.")
            else:
                print(f"Collection '{self.weaviate_class_name}' already exists and is compatible. Skipping creation.")
                self._ensure_schema_properties()

        except Exception as e:
            print(f"ERROR: Failed to create or check Weaviate schema: {e}")

    def _ensure_schema_properties(self):
        """Add properties introduced after a collection was first created."""
        wanted = {
            "chunk_index": DataType.INT,
            "section_category": DataType.TEXT,
            "study_id": DataType.TEXT,
            "doc_role": DataType.TEXT,
            "concept_ids": DataType.TEXT_ARRAY,
            "entity_aliases": DataType.TEXT_ARRAY,
            "x0": DataType.NUMBER,
            "y0": DataType.NUMBER,
            "x1": DataType.NUMBER,
            "y1": DataType.NUMBER,
        }
        try:
            collection = self.weaviate_client.collections.get(self.weaviate_class_name)
            config = collection.config.get()
            prop_names = {prop.name for prop in (config.properties or [])}
            for name, data_type in wanted.items():
                if name not in prop_names:
                    collection.config.add_property(Property(name=name, data_type=data_type))
                    print(f"Added '{name}' to '{self.weaviate_class_name}'. Re-ingest this PDF so existing objects receive a value.")
        except Exception as e:
            print(f"WARNING: Could not ensure schema properties: {e}")

    def _vertex_embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Embed a batch through Vertex AI using Application Default Credentials."""
        import google.auth
        from google.auth.transport.requests import Request as AuthRequest
        import requests

        credentials, detected_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(AuthRequest())
        project = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or detected_project
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        model = embedding_model_name()
        if not project:
            raise EmbeddingGenerationError("Set VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT for Vertex embeddings.")
        host = "https://aiplatform.googleapis.com" if location == "global" else f"https://{location}-aiplatform.googleapis.com"
        url = (
            f"{host}/v1/projects/{project}/locations/{location}"
            f"/publishers/google/models/{model}:predict"
        )
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {credentials.token}"},
            json={"instances": [{"content": text, "task_type": task_type} for text in texts]},
            timeout=60,
        )
        if response.status_code >= 400:
            raise EmbeddingGenerationError(
                f"Vertex embedding {model} returned {response.status_code}: {response.text[:300]}"
            )
        predictions = response.json().get("predictions") or []
        vectors = []
        for prediction in predictions:
            values = (prediction.get("embeddings") or {}).get("values")
            if not isinstance(values, list) or not values:
                raise EmbeddingGenerationError("Vertex embedding response did not contain a vector.")
            vectors.append(values)
        if len(vectors) != len(texts):
            raise EmbeddingGenerationError(
                f"Vertex returned {len(vectors)} vectors for {len(texts)} texts."
            )
        return vectors

    def _get_embeddings_from_api(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """Gets embeddings in batches of up to 100 from Vertex text-embedding-004."""
        if not texts:
            return []

        embeddings: List[List[float]] = []
        model = embedding_model_name()
        batch_size = self._EMBED_BATCH_SIZE
        print(f"Getting {len(texts)} Vertex embeddings using model: {model} (batch size {batch_size})")

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            batch_embeddings = None
            retries = 5
            for attempt in range(retries):
                try:
                    batch_embeddings = self._vertex_embed(batch, task_type)
                    break
                except Exception as e:
                    if not _should_retry_embedding(e) or attempt == retries - 1:
                        if isinstance(e, EmbeddingGenerationError) and not _should_retry_embedding(e):
                            raise
                        print(f"Error calling Vertex embeddings for batch starting at index {start}: {e}")
                        break
                    wait = min(60.0, (2 ** attempt) + random.uniform(0, 1))
                    print(f"Retrying in {wait:.1f} seconds... (Attempt {attempt + 2}/{retries})")
                    time.sleep(wait)

            if batch_embeddings is None:
                failed = len(texts) - len(embeddings)
                raise EmbeddingGenerationError(
                    f"Failed to get embeddings for {failed} out of {len(texts)} texts after multiple retries."
                )

            embeddings.extend(batch_embeddings)
            print(f"  Processed {len(embeddings)}/{len(texts)} texts...")

        return embeddings

    def _index_chunks_in_weaviate(self):
        """Indexes chunks using Native Weaviate Batch with User-Provided Embeddings."""
        if not self.weaviate_client or not self.weaviate_client.is_ready():
            print("ERROR: Weaviate client is not ready in _index_chunks_in_weaviate.")
            return

        print(f"Indexing {len(self.all_documents)} chunks...")

        # 1. Generate Embeddings via API
        texts = [text_to_embed(doc) for doc in self.all_documents]
        embeddings = self._get_embeddings_from_api(texts)

        if not embeddings or len(embeddings) != len(self.all_documents):
            print("Error: Document/Embedding count mismatch or failed to get embeddings.")
            return

        collection = self.weaviate_client.collections.get(self.weaviate_class_name)

        try:
            with collection.batch.dynamic() as batch:
                for i, doc in enumerate(self.all_documents):
                    # 2. Add object with Manual Vector
                    properties = {
                        "text": doc.page_content,
                        "source": doc.metadata.get("source", self.normalized_pdf_url),
                        "page_number": doc.metadata.get("page_number"),
                        "element_type": doc.metadata.get("element_type"),
                        "token_size": doc.metadata.get("token_size"),
                        "chunk_index": doc.metadata.get("chunk_index", i),
                        "section_category": doc.metadata.get("section_category", "Front matter"),
                        "study_id": doc.metadata.get("study_id") or "",
                        "doc_role": doc.metadata.get("doc_role") or "main",
                        "concept_ids": _text_list(doc.metadata.get("concept_ids")),
                        "entity_aliases": _text_list(doc.metadata.get("entity_aliases")),
                    }
                    properties.update(bbox_properties(doc.metadata))
                    batch.add_object(
                        properties=properties,
                        vector=embeddings[i]
                    )

            if len(collection.batch.failed_objects) > 0:
                print(f"Failed to import {len(collection.batch.failed_objects)} objects.")
            else:
                print("Successfully indexed all chunks.")

        except Exception as e:
            print(f"Error indexing chunks: {e}")

        # KILL PIPELINE RAM: Explicitly delete the heavy vectors from memory
        del embeddings
        del texts
        gc.collect()

    def _check_weaviate_for_data(self) -> bool:
        """Checks if data exists using Native Client."""
        if not self.weaviate_client or not self.weaviate_client.is_ready():
            print("ERROR: Weaviate client is not ready in _check_weaviate_for_data.")
            return False
        try:
            collection = self.weaviate_client.collections.get(self.weaviate_class_name)
            # Just check if any object exists in this specific collection
            response = collection.query.fetch_objects(limit=1)
            return len(response.objects) > 0
        except Exception:
            return False

    def _load_data_from_weaviate(self):
        """Loads chunks using Native Client (gRPC fetch)."""
        print("Loading data via Native Client...")
        if not self.weaviate_client or not self.weaviate_client.is_ready():
            print("ERROR: Weaviate client is not ready in _load_data_from_weaviate.")
            return
            
        try:
            collection = self.weaviate_client.collections.get(self.weaviate_class_name)
            
            # === FIX: Removed the filter by 'source'. 
            # Since the Class Name is the Hash, we want everything in this Class.
            response = collection.query.fetch_objects(
                limit=10000
            )
            
            docs = []
            for obj in response.objects:
                metadata = metadata_from_properties(obj.properties)
                metadata["weaviate_id"] = str(obj.uuid)
                docs.append(Document(page_content=obj.properties['text'], metadata=metadata))

            docs.sort(key=lambda doc: (
                doc.metadata.get("chunk_index") is None,
                doc.metadata.get("chunk_index") if doc.metadata.get("chunk_index") is not None else 0,
                doc.metadata.get("page_number") or 0,
            ))
            self.all_documents = docs
            print(f"Successfully loaded {len(self.all_documents)} documents.")
        except Exception as e:
            print(f"ERROR: Failed to load data: {e}")

    def persist_study_tags(self, documents) -> int:
        """Write study_id and doc_role onto chunks that are already stored.

        Bundle tagging happens after the first index, so this updates properties
        only. The stored vectors stay in place.
        """
        docs = [doc for doc in (documents or []) if isinstance(getattr(doc, "metadata", None), dict)]
        if not docs:
            return 0
        client = getattr(self, "weaviate_client", None)
        class_name = (getattr(self, "weaviate_class_name", "") or "").strip()
        if client is None or not class_name:
            print("ERROR: Weaviate client is not ready in persist_study_tags.")
            return 0
        try:
            ready = client.is_ready()
        except Exception:
            ready = False
        if not ready or not client.collections.exists(class_name):
            print(f"ERROR: Collection '{class_name}' is not available. Study tags were not saved.")
            return 0

        self._ensure_schema_properties()
        collection = client.collections.get(class_name)
        uuid_by_index = {}
        uuid_by_text = {}
        if any(not doc.metadata.get("weaviate_id") for doc in docs):
            response = collection.query.fetch_objects(limit=10000)
            for obj in response.objects:
                props = obj.properties or {}
                raw_index = props.get("chunk_index")
                if raw_index is not None:
                    uuid_by_index[int(raw_index)] = str(obj.uuid)
                text = props.get("text") or ""
                if text and text not in uuid_by_text:
                    uuid_by_text[text] = str(obj.uuid)

        updated = 0
        for doc in docs:
            meta = doc.metadata
            uid = str(meta.get("weaviate_id") or "")
            if not uid:
                raw_index = meta.get("chunk_index")
                if raw_index is not None:
                    uid = uuid_by_index.get(int(raw_index), "")
                if not uid:
                    uid = uuid_by_text.get(doc.page_content or "", "")
            if not uid:
                continue
            meta["weaviate_id"] = uid
            try:
                collection.data.update(
                    uuid=uid,
                    properties={
                        "study_id": meta.get("study_id") or "",
                        "doc_role": meta.get("doc_role") or "main",
                    },
                )
            except Exception as exc:
                print(f"WARNING: Could not update study tags for {uid}: {exc}")
                continue
            updated += 1
        print(f"Updated study tags on {updated} of {len(docs)} chunks in '{class_name}'.")
        return updated

    def _perform_hybrid_search(
        self,
        query: str,
        k: int = 20,
        exclude_references: bool = True,
        study_id: str = "",
        doc_roles: List[str] | None = None,
        collection_name: str = "",
    ) -> List[Document]:
        """
        Hybrid search with reciprocal rank fusion.
        Bibliography chunks are left out unless exclude_references is False.
        Optionally filters by study_id and/or doc_role values (e.g., ["supplement", "protocol"]).
        """
        print("Executing Hybrid Search (RRF)...")
        if not self.weaviate_client or not self.weaviate_client.is_ready():
            print("ERROR: Weaviate client is not ready in _perform_hybrid_search.")
            return []

        class_name = (collection_name or self.weaviate_class_name or "").strip()
        if collection_name and not self.weaviate_client.collections.exists(class_name):
            print(f"Companion collection '{class_name}' is not in Weaviate. Skipping that hop.")
            return []
        collection = self.weaviate_client.collections.get(class_name)
        conditions = []
        if exclude_references:
            conditions.append(
                Filter.any_of([
                    Filter.by_property("section_category").not_equal("References"),
                    Filter.by_property("section_category").is_none(True),
                ])
            )
        if study_id:
            conditions.append(Filter.by_property("study_id").equal(study_id))
        if doc_roles:
            role_filters = [Filter.by_property("doc_role").equal(role) for role in doc_roles if role]
            if role_filters:
                combined_roles = role_filters[0]
                for rf in role_filters[1:]:
                    combined_roles = combined_roles | rf
                conditions.append(combined_roles)

        search_filter = None
        if len(conditions) == 1:
            search_filter = conditions[0]
        elif len(conditions) > 1:
            search_filter = Filter.all_of(conditions)

        # 1. Generate the Query Vector via API
        query_embeddings = self._get_embeddings_from_api([query], task_type="RETRIEVAL_QUERY")
        if not query_embeddings:
            return []

        query_vector = query_embeddings[0]

        try:
            # 2. Execute Hybrid Query
            response = collection.query.hybrid(
                query=query,
                vector=query_vector,
                fusion_type=HybridFusion.RANKED,
                limit=k,
                filters=search_filter,
                return_metadata=["score"],
            )

            docs = []
            for obj in response.objects:
                metadata = metadata_from_properties(obj.properties)
                docs.append(Document(page_content=obj.properties["text"], metadata=metadata))

            return docs

        except Exception as e:
            print(f"Error during native search: {e}")
            return []
