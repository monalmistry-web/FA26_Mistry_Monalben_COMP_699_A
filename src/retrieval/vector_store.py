from langchain_chroma import Chroma

from ingestion.embedding import get_embedding_model

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
    
