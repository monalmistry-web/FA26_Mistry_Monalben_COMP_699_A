from unittest.mock import patch, MagicMock

from retrieval.vector_store import (create_vector_store, add_chunks_to_vector_store, search_similar_chunks)

def test_vector_store():
    """ Test that the ChromaDB vector store is created."""

    with patch("retrieval.vector_store.get_embedding_model")as mock_embedding:
        with patch("retrieval.vector_store.Chroma")as mock_chroma:

            create_vector_store()
            mock_embedding.assert_called_once()
            mock_chroma.assert_called_once()

def test_add_chunks_to_vector_store():
    """ Test that chunks and metadata are stored."""

    vector_store = MagicMock()

    chunks = [
        {"text": "Introduction to AI", "page": 1},
        {"text": "Machine learning concepts", "page": 2}
    ]

    source_id = "paper_123"

    add_chunks_to_vector_store(vector_store,chunks,source_id)

    vector_store.delete.assert_called_once_with(where={"source_id": source_id})

    vector_store.add_texts.assert_called_once_with(
        texts = [
             "Introduction to AI",
            "Machine learning concepts",
        ],
        metadatas=[
              {"page": 1, "source_id": source_id},
            {"page": 2, "source_id": source_id},
        ],
        ids=[
             "paper_123_0",
            "paper_123_1",
        ],
    )

def test_search_similar_chunks():
    """ Test that relevant chunks are retrived for a PDF."""
    vector_store = MagicMock()
    expected_results=[MagicMock(),MagicMock()]

    vector_store.similarity_search.return_value = expected_results

    results= search_similar_chunks(
        vector_store,
        "What is artificial intelligence?",
        "paper_123",
        k=3,
    )
    vector_store.similarity_search.assert_called_once_with(
    "What is artificial intelligence?",
    k=3,
    filter={"source_id": "paper_123"},
    )
    assert results == expected_results

def test_add_empty_chunks_to_vector_store():
    """Test that empty chunks are handled without errors."""

    vector_store = MagicMock()

    add_chunks_to_vector_store(
        vector_store,
        [],
        "paper_123",
    )

    vector_store.add_texts.assert_called_once_with(
        texts=[],
        metadatas=[],
        ids=[],
    )

