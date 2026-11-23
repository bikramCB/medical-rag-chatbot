import gc
import os
import random
import re
import time
from typing import List

from langchain.schema import Document
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.query import Filter, HybridFusion


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
    """Tables are embedded from their two-sentence summary. The stored text stays the full table."""
    summary = doc.metadata.get("embed_text")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return doc.page_content


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
            

    def _perform_hybrid_search(self, query: str, k: int = 20, exclude_references: bool = True) -> List[Document]:
        """
        Hybrid search with reciprocal rank fusion.
        Bibliography chunks are left out unless exclude_references is False.
        """
        print("Executing Hybrid Search (RRF)...")
        if not self.weaviate_client or not self.weaviate_client.is_ready():
            print("ERROR: Weaviate client is not ready in _perform_hybrid_search.")
            return []
            
        collection = self.weaviate_client.collections.get(self.weaviate_class_name)
        # Missing section_category (papers indexed before this field) still matches.
        search_filter = None
        if exclude_references:
            search_filter = Filter.any_of([
                Filter.by_property("section_category").not_equal("References"),
                Filter.by_property("section_category").is_none(True),
            ])

        # 1. Generate the Query Vector via API
        query_embeddings = self._get_embeddings_from_api([query], task_type="RETRIEVAL_QUERY")
        if not query_embeddings:
            return []
        
        query_vector = query_embeddings[0]

        try:
            # 2. Execute Hybrid Query
            response = collection.query.hybrid(
                # Keyword Component
                query=query,
                
                # Vector Component
                vector=query_vector,
                
                # Fusion Component (Reciprocal Rank Fusion)
                fusion_type=HybridFusion.RANKED, 
                
                limit=k,
                filters=search_filter,
                return_metadata=["score"]
            )
            
            docs = []
            for obj in response.objects:
                metadata = metadata_from_properties(obj.properties)
                docs.append(Document(page_content=obj.properties["text"], metadata=metadata))
            
            return docs

        except Exception as e:
            print(f"Error during native search: {e}")
            return []
