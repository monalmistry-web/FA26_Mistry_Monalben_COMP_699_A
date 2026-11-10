from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_pages_into_chunks(pages):
    """Split PDF pages into chunks while preserving page numbers."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "text":chunk,
                "page":page["page"],
            })
    return chunks