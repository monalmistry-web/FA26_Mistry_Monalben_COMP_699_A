import hashlib
import re
import time
from itertools import zip_longest

import streamlit as st

from generation.answer_generator import (
    generate_answer,
    validate_answer_citations,
)
from ingestion.pdf_loader import extract_pages_from_pdf
from ingestion.text_splitter import split_pages_into_chunks
from retrieval.vector_store import (
    add_chunks_to_vector_store,
    create_vector_store,
    expand_with_neighbor_chunks,
    search_hybrid_chunks,
    search_mmr_chunks,
    search_similar_chunks,
    split_question_into_queries,
)


def is_reference_list_chunk(text):
    """Identify chunks containing multiple bibliography entries."""
    reference_entries = re.findall(
        r"\[\d{1,3}\]\s+[A-Z][A-Za-zÀ-ÿ.'’-]*\s+[A-Z]",
        text
    )

    return len(reference_entries) >= 2

st.set_page_config(page_title="Academic Research Paper Assistant",page_icon=":books:",layout="wide")

st.title("Academic Research Paper Assistant")
st.write("Upload your research paper in PDF format to extract text and analyze it.")

uploaded_file = st.file_uploader("Choose a research paper",type=["pdf"])

if uploaded_file is not None:
    try:
        page_count,pages = extract_pages_from_pdf(uploaded_file)
        source_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        
        st.success("PDF opened successfully!")
        st.write(f"**Number of pages:** {page_count}")

        full_text = "\n\n".join(page["text"] for page in pages)

        chunks = []

        if full_text.strip():
            st.text_area("Extracted Text", value=full_text, height=300)
            chunks = split_pages_into_chunks(pages)
            st.write(f"**Number of chunks:** {len(chunks)}")
        else:
            st.warning("No selectable text was found." 
                       "The PDF may contain scanned pages.")
        if st.button("Process and Store Research Paper"):
            if not chunks:
                st.warning("No text chunks were available to store.")
            else:
                with st.spinner("Generating embeddings and storing chunks..."):
                    vector_store = create_vector_store()
                    add_chunks_to_vector_store(vector_store,chunks,source_id)
                st.success("Research paper processed and stored successfully!")

        st.subheader("Search Your Research Paper")
        question = st.text_input("Enter a question about your research paper:")

        retrieval_method = st.radio("Select retrieval method:",["Similarity Search", "MMR", "Hybrid Search"],horizontal=True,)

        if st.button("Search"):
            if question.strip():
                with st.spinner("Searching for relevant passages..."):
                    retrieval_start = time.perf_counter()
                    vector_store = create_vector_store()
                    queries = split_question_into_queries(question)

                    results = []
                    seen_chunks = set()
                    query_results = []

                    # Retrieve up to 4 passages per query.
                    k_per_query = 4

                    for query in queries:
                        if retrieval_method == "Similarity Search":
                            retrieved = search_similar_chunks(
                                vector_store, query, source_id, k=k_per_query)
                        elif retrieval_method == "MMR":
                            retrieved = search_mmr_chunks(
                                vector_store, query, source_id, k=k_per_query)
                        else:
                            retrieved = search_hybrid_chunks(
                                vector_store, query, source_id, k=k_per_query)
                        retrieved = [
                            doc for doc in retrieved
                            if not is_reference_list_chunk(doc.page_content)]
                        expanded = expand_with_neighbor_chunks(
                            vector_store, retrieved, source_id, max_chunks=20
                        )
                        query_results.append(expanded)
                retrieval_time = time.perf_counter() - retrieval_start
                st.write(f"Retrieval time: {retrieval_time:.2f} seconds")

                if query_results:
                    for group in zip_longest(*query_results):
                        for result in group:
                            if result is None:
                                continue
                            chunk_key = (
                                result.metadata.get("page"),
                                result.page_content,
                            )
                            if chunk_key not in seen_chunks:
                                seen_chunks.add(chunk_key)
                                results.append(result)
                            if len(results) >= 20:
                                break
                        if len(results) >= 20:
                            break

                    # Display the retrieved passages.
                    for index, result in enumerate(results, start=1):
                        st.write(f"**Result {index} - Page {result.metadata['page']}**")

                        st.write(result.page_content)
                    # Prepare context for the language model.
                    context_results = [
                        result for result in results
                        if not is_reference_list_chunk(result.page_content)]

                    # Preserve retrieval ranking rather than grouping passages by page.
                    if not context_results:
                         st.warning(
                              "No usable passages were found after filtering reference entries.")
                         st.stop()
                    context = "\n\n".join(
                        f"[Page {result.metadata['page']}]\n{result.page_content}"
                        for result in context_results
)

                    # Generate an answer using the retrived context.
                    valid_pages = {
                        result.metadata["page"]
                        for result in context_results
                    }

                    with st.spinner("Generating answer..."):
                        start_time = time.perf_counter()

                        answer = generate_answer(question, context, valid_pages)

                        generation_time = time.perf_counter() - start_time

                    st.write(f"Answer generation time: {generation_time:.2f} seconds")

                    st.subheader("Generated Answer")
                    st.write(answer)

                    citation_check = validate_answer_citations(answer, valid_pages)

                    if not citation_check["has_citations"]:
                         st.warning("Citation validation: The generated answer " "does not contain PDF page citations.")
                    elif citation_check["invalid_pages"]:
                         st.warning("Citation validation: The answer cites pages " "that were not in the retrieved context: " f"{citation_check['invalid_pages']}")
                    else:
                        st.info("Citation validation: All detected page citations " "refer to retrieved PDF pages.")
                    
                else:
                    st.info("No relevant passages were found.")
            else:
                st.warning("Please enter a question.")

    except Exception as error:
        st.error(str(error))
