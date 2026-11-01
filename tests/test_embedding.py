from unittest.mock import patch

from ingestion.embedding import get_embedding_model

def test_embedding_model():
    """ Test that the local embedding model is created"""

    with patch("ingestion.embedding.OllamaEmbeddings")as mock_embeddings:
        get_embedding_model()

        mock_embeddings.assert_called_once_with(
            model="nomic-embed-text"
        )