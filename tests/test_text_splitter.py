from ingestion.text_splitter import split_pages_into_chunks

def test_split_pages_into_chunks():
    """ Test that PDF page text is split into chunks with page numbers."""

    pages = [
        {
            "page": 1,
            "text": "This is a sample reasearch paper." * 100,
        }
    ]

    chunks = split_pages_into_chunks(pages)

    #Check that chunks were created.
    assert len(chunks) > 1

    #Check that every chunk contains text.
    assert all(chunk["text"] for chunk in chunks)

    #Check that page numbers are preserved.
    assert all(chunk["page"] == 1 for chunk in chunks)

def test_split_pages_into_chunks_with_empty_pages():
    """Test that empty input returns an empty list."""

    pages = []

    chunks = split_pages_into_chunks(pages)

    assert chunks == []