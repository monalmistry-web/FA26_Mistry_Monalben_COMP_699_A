from langchain_ollama import OllamaEmbeddings

def get_embedding_model():
    """ create and return the local embedding model."""
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text")

    return embeddings