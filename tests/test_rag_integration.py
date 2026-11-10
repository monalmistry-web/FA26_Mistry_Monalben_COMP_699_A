from unittest.mock import patch

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from retrieval.vector_store import(add_chunks_to_vector_store,search_similar_chunks)
from generation.answer_generator import generate_answer

class FakeEmbeddings(Embeddings):
    """ Provide simple embeddings for integration testing."""

    def embed_documents(self,texts):
        return [[1.0,0.0] for _ in texts]

    def embed_query(self,text):
        return [1.0,0.0]

def test_retrieval_and_answer_generation_work_together(tmp_path):
    # 1. create a temporary vector store
    vector_store = Chroma(
        collection_name="test_reasearch_papers",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path),

    )

    # 2. Prepare a sample research paper passage.
    chunks = [
        {
            "text":("The two dominant areas of human responsibility "
                    "are orchestration and verification."),
                    "page": 2,
        }
    ]

    source_id="test-paper"

    # 3. Store the research paper passage.

    add_chunks_to_vector_store(
        vector_store,chunks,source_id,
        )

    # 4. Retrieve the relevant passage.

    question = "What are the two areas of human responsibility?"

    results = search_similar_chunks(
        vector_store,question,source_id,k=1,)

    assert len(results) == 1
    assert results[0].metadata["page"] == 2

    # 5. Prepare the retrieved passage as context.
    context = "\n\n".join(
        f"[Page {result.metadata['page']}]\n{result.page_content}"
        for result in results
    )

    # 6. Mock the language model response.
    with patch("generation.answer_generator.ChatOllama") as mock_chat:
        mock_llm = mock_chat.return_value
        mock_llm.invoke.return_value.content = (
            "The two areas are orchestration and verification [Page 2]."
        )
        answer = generate_answer(question,context)

    # 7. Verify answer generation.
    assert "orchestration" in answer.lower()
    assert "verification" in answer.lower()

    mock_llm.invoke.assert_called_once()
