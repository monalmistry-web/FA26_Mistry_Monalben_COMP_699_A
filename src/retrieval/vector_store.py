from langchain_chroma import Chroma

from ingestion.embedding import get_embedding_model
import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

def create_vector_store():
    """Create or connect to local ChromaDB vector store."""
    embeddings = get_embedding_model()

    vector_store = Chroma (
        collection_name="research_papers",
        embedding_function=embeddings,
        persist_directory="data/chroma",
    )

    return vector_store


def add_chunks_to_vector_store(vector_store, chunks, source_id):
    """Store PDF chunks and thier page metadeta in ChromaDB."""

    # Remove previously stored chunks for this PDF.
    vector_store.delete(where={"source_id": source_id})
    texts = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "page": chunk["page"],
            "source_id": source_id,
            } 
            for chunk in chunks
    ]
    ids=[
        f"{source_id}_{index}"
        for index in range(len(chunks))
        ]
    
    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
    )

def search_similar_chunks(vector_store, question, source_id, k=3):
    """Find the most relevant chunks for a question"""
    results = vector_store.similarity_search(
        question,
        k=k,
        filter={"source_id":source_id},

    )
    return results


def search_mmr_chunks(vector_store, question, source_id, k=5):
    """ Find relevant and diverse chunks using MMR retrieval."""
    results = vector_store.max_marginal_relevance_search(
        question,
        k=k,
        fetch_k=20,
        lambda_mult=0.5,
        filter={"source_id": source_id},
    )
    return results


def split_question_into_queries(question):
    """Split multipart questions while preserving the shared topic."""

    parts = re.split(
        r",?\s+\band\b\s+(?=(?:what|which|how|why|where|when|who)\b)",
        question.strip(),
        flags=re.IGNORECASE,
    )

    queries = [" ".join(part.split()) for part in parts if part.strip()]

    if len(queries) <= 1:
        return queries

    # Find the topic mentioned in the first question.
    topic_match = re.search(
        r"\b(?:of|for|about|on)\s+([^,?]+)$",
        queries[0],
        flags=re.IGNORECASE,
    )

    if topic_match:
        topic = topic_match.group(1).strip()

        for i in range(1, len(queries)):
            if topic.lower() not in queries[i].lower():
                queries[i] = (
                    queries[i].rstrip("?")
                    + f" about {topic}?"
                )

    return queries


def search_keyword_chunks(vector_store, question, source_id, k=5):
    """Retrieve relevant PDF chunks using BM25 keyword ranking."""

    stored = vector_store.get(
        where={"source_id": source_id},
        include=["documents", "metadatas"],
    )

    documents = stored["documents"]
    metadatas = stored["metadatas"]

    if not documents:
        return []

    def tokenize(text):
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were",
            "what", "which", "how", "why", "who",
            "do", "does", "did", "and", "or", "of",
            "to", "in", "on", "for", "with", "from",
            "by", "their", "its", "it", "this", "that",
    }
        words = re.findall(r"\b\w+\b", text.lower())
        return [word for word in words if word not in stopwords]

    tokenized_documents = [tokenize(text) for text in documents]
    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    bm25 = BM25Okapi(tokenized_documents)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    return [
        Document(
            page_content=documents[index],
            metadata=metadatas[index] or {},
        )
        for index in ranked_indices[:k]
        if scores[index] > 0
    ]

def search_hybrid_chunks(vector_store, question, source_id, k=5):
    """Combine semantic and BM25 retrieval using reciprocal rank fusion."""

    semantic_results = search_similar_chunks(
        vector_store, question, source_id, k=10
    )

    keyword_results = search_keyword_chunks(
        vector_store, question, source_id, k=10
    )

    scores = {}
    documents = {}

    for results in [semantic_results, keyword_results]:
        for rank, document in enumerate(results, start=1):
            key = (
                document.metadata.get("page"),
                document.page_content,
            )

            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            documents[key] = document

    ranked_keys = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True,
    )

    return [documents[key] for key in ranked_keys[:k]]

def expand_with_neighbor_chunks(vector_store, results, source_id, max_chunks=10):
    """Keep retrieved chunks first, then add neighbors if space remains."""

    if not results or max_chunks <= 0:
        return []

    stored = vector_store.get(
        where={"source_id": source_id},
        include=["documents", "metadatas"],
    )

    chunks_by_id = {
        chunk_id: (text, metadata)
        for chunk_id, text, metadata in zip(
            stored["ids"],
            stored["documents"],
            stored["metadatas"],
        )
    }

    expanded = []
    seen = set()
    matched_indices = []

    # First, preserve the retrieved results in their ranking order.
    for result in results:
        matching_id = next(
            (
                chunk_id
                for chunk_id, (text, metadata) in chunks_by_id.items()
                if text == result.page_content
                and metadata.get("page") == result.metadata.get("page")
            ),
            None,
        )

        if matching_id is None:
            continue

        if matching_id not in seen:
            text, metadata = chunks_by_id[matching_id]
            expanded.append(Document(page_content=text, metadata=metadata))
            seen.add(matching_id)
            matched_indices.append(int(matching_id.rsplit("_", 1)[1]))

        if len(expanded) >= max_chunks:
            return expanded

    # Then add neighboring chunks using remaining space.
    for chunk_index in matched_indices:
        for index in [chunk_index - 1, chunk_index + 1]:
            if len(expanded) >= max_chunks:
                return expanded

            neighbor_id = f"{source_id}_{index}"

            if neighbor_id in chunks_by_id and neighbor_id not in seen:
                text, metadata = chunks_by_id[neighbor_id]
                expanded.append(
                    Document(page_content=text, metadata=metadata)
                )
                seen.add(neighbor_id)

    return expanded
