
from unittest.mock import MagicMock

from langchain_core.documents import Document

from retrieval.vector_store import expand_with_neighbor_chunks


def test_expands_neighboring_chunks():
    """Retrieved chunks should include their immediate neighbors."""

    source_id = "test_paper"

    vector_store = MagicMock()
    vector_store.get.return_value = {
        "ids": [
            "test_paper_0",
            "test_paper_1",
            "test_paper_2",
        ],
        "documents": [
            "Introduction",
            "Main research findings",
            "Conclusion",
        ],
        "metadatas": [
            {"page": 1, "source_id": source_id},
            {"page": 2, "source_id": source_id},
            {"page": 3, "source_id": source_id},
        ],
    }

    retrieved = [
        Document(
            page_content="Main research findings",
            metadata={"page": 2, "source_id": source_id},
        )
    ]

    results = expand_with_neighbor_chunks(
        vector_store,
        retrieved,
        source_id,
        max_chunks=8,
    )

    assert [doc.page_content for doc in results] == [
        "Main research findings",
        "Introduction",
        "Conclusion",
    ]


def test_neighbor_expansion_avoids_duplicates():
    """Chunks already retrieved should not be added twice."""

    source_id = "test_paper"
    vector_store = MagicMock()

    vector_store.get.return_value = {
        "ids": [
            "test_paper_0",
            "test_paper_1",
            "test_paper_2",
        ],
        "documents": [
            "Introduction",
            "Main research findings",
            "Conclusion",
        ],
        "metadatas": [
            {"page": 1, "source_id": source_id},
            {"page": 2, "source_id": source_id},
            {"page": 3, "source_id": source_id},
        ],
    }

    retrieved = [
        Document(
            page_content="Main research findings",
            metadata={"page": 2, "source_id": source_id},
        ),
        Document(
            page_content="Conclusion",
            metadata={"page": 3, "source_id": source_id},
        ),
    ]

    results = expand_with_neighbor_chunks(
        vector_store,
        retrieved,
        source_id,
        max_chunks=8,
    )

    contents = [doc.page_content for doc in results]

    assert contents == [
        "Main research findings",
        "Conclusion",
        "Introduction",
    ]

    assert len(contents) == len(set(contents))


def test_neighbor_expansion_respects_max_chunks():
    """Expansion should not exceed the maximum number of chunks."""

    source_id = "test_paper"
    vector_store = MagicMock()

    vector_store.get.return_value = {
        "ids": [f"{source_id}_{i}" for i in range(5)],
        "documents": [f"Chunk {i}" for i in range(5)],
        "metadatas": [
            {"page": i + 1, "source_id": source_id}
            for i in range(5)
        ],
    }

    retrieved = [
        Document(
            page_content="Chunk 2",
            metadata={"page": 3, "source_id": source_id},
        )
    ]

    results = expand_with_neighbor_chunks(
        vector_store,
        retrieved,
        source_id,
        max_chunks=2,
    )

    assert len(results) == 2

    assert [doc.page_content for doc in results] == [
        "Chunk 2",
        "Chunk 1",
    ]


def test_neighbor_expansion_filters_by_source():
    """Only chunks belonging to the selected PDF should be retrieved."""

    source_id = "test_paper"
    vector_store = MagicMock()

    vector_store.get.return_value = {
        "ids": [
            "test_paper_0",
            "test_paper_1",
        ],
        "documents": [
            "Introduction",
            "Main research findings",
        ],
        "metadatas": [
            {"page": 1, "source_id": source_id},
            {"page": 2, "source_id": source_id},
        ],
    }

    retrieved = [
        Document(
            page_content="Main research findings",
            metadata={"page": 2, "source_id": source_id},
        )
    ]

    results = expand_with_neighbor_chunks(
        vector_store,
        retrieved,
        source_id,
        max_chunks=8,
    )

    vector_store.get.assert_called_once_with(
        where={"source_id": source_id},
        include=["documents", "metadatas"],
    )

    assert all(
        doc.metadata["source_id"] == source_id
        for doc in results
    )
