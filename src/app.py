import hashlib
import re
import time
from itertools import zip_longest
from pathlib import Path

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
    reference_entries = re.findall(r"\[\d{1,3}\]\s+[A-Z][A-Za-zÀ-ÿ.'’-]*\s+[A-Z]", text)

    return len(reference_entries) >= 2


st.set_page_config(
    page_title="Academic Research Paper Assistant",
    page_icon="📚",
    layout="wide",
)
# Load custom application styling.
css_path = Path(__file__).resolve().parent / "styles.css"

st.markdown(
    f"<style>{css_path.read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)
# Application header.
st.html(
    """
    <div class="workspace-header">
        <h1>📖 Academic Research Paper Assistant</h1>
        <p>AI-powered question answering for academic research papers using RAG.</p>
        <div class="workspace-features">
            <span>◈ Local AI</span>
            <span>◈ RAG Architecture</span>
            <span>◈ Source Citations</span>
        </div>
    </div>
    """
)
# Main workspace layout.
document_col, question_col = st.columns(
    [2, 3],
    gap="medium",
)
with document_col:
    st.subheader("📂 Research Document")

    uploaded_file = st.file_uploader(
        "Upload research paper",
        type=["pdf"],
        help="Upload a PDF containing selectable text.",
    )

    st.caption("PDF files only · Selectable text required")
# Display guidance before a research paper is uploaded.
if uploaded_file is None:
    with question_col:
        st.subheader("💬 Ask AI Assistant")

        st.caption(
            "Ask questions about your research paper "
            "and receive evidence-based answers."
        )

        st.info("Upload a research paper to get started.")

        st.markdown("**How it works**")

        st.markdown(
            """
            1. Upload an academic research paper.
            2. Process the research paper.
            3. Enter your research question.
            4. Select a retrieval method.
            5. Generate an answer with PDF page citations.
            """
        )
if uploaded_file is not None:
    try:
        page_count, pages = extract_pages_from_pdf(uploaded_file)
        source_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

        with document_col:
            st.success("PDF opened successfully!")

            full_text = "\n\n".join(page["text"] for page in pages)

            chunks = []

            if full_text.strip():
                chunks = split_pages_into_chunks(pages)

                st.markdown("**Document Overview**")
                st.caption(uploaded_file.name)

                col1, col2 = st.columns(2)

                col1.metric("PDF Pages", page_count)
                col2.metric("Text Chunks", len(chunks))

                st.caption(f"Document size: {uploaded_file.size / 1000:.1f} KB")

            else:
                st.warning(
                    "No selectable text was found. The PDF may contain scanned pages."
                )

            if st.button(
                "Process Research Paper",
                type="primary",
                use_container_width=True,
                disabled=not chunks,
                help="Generate embeddings and store the paper for retrieval.",
            ):
                if not chunks:
                    st.warning("No text chunks were available to store.")
                else:
                    with st.spinner(
                        "Processing research paper and generating embeddings..."
                    ):
                        vector_store = create_vector_store()

                        add_chunks_to_vector_store(
                            vector_store,
                            chunks,
                            source_id,
                        )

                    st.success("Research paper processed successfully!")

            if full_text.strip():
                with st.expander("View Extracted Text"):
                    st.text_area(
                        "Extracted PDF Text",
                        value=full_text,
                        height=300,
                        disabled=True,
                        label_visibility="collapsed",
                    )

        with question_col:
            st.subheader("💬 Ask AI Assistant")

            st.caption(
                "Ask questions about your research paper "
                "and receive evidence-based answers."
            )

            question = st.text_area(
                "Research Question",
                placeholder=("e.g., What are the main findings of this study?"),
                height=120,
            )

            retrieval_method = st.radio(
                "Retrieval Method",
                ["Similarity Search", "MMR", "Hybrid Search"],
                horizontal=False,
            )

            retrieval_descriptions = {
                "Similarity Search": (
                    "Finds passages that are semantically similar to your question."
                ),
                "MMR": (
                    "Balances relevance and diversity to reduce "
                    "redundant retrieved passages."
                ),
                "Hybrid Search": (
                    "Combines semantic and keyword-based retrieval "
                    "to find relevant passages."
                ),
            }

            st.info(retrieval_descriptions[retrieval_method])

            generate_clicked = st.button(
                "Generate Answer",
                type="primary",
                use_container_width=True,
            )

        if generate_clicked:
            if question.strip():
                with st.spinner(
                    "Retrieving relevant passages from your research paper..."
                ):
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
                                vector_store, query, source_id, k=k_per_query
                            )
                        elif retrieval_method == "MMR":
                            retrieved = search_mmr_chunks(
                                vector_store, query, source_id, k=k_per_query
                            )
                        else:
                            retrieved = search_hybrid_chunks(
                                vector_store, query, source_id, k=k_per_query
                            )
                        retrieved = [
                            doc
                            for doc in retrieved
                            if not is_reference_list_chunk(doc.page_content)
                        ]
                        expanded = expand_with_neighbor_chunks(
                            vector_store, retrieved, source_id, max_chunks=20
                        )
                        query_results.append(expanded)
                retrieval_time = time.perf_counter() - retrieval_start

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

                    # Prepare context for the language model.
                    context_results = [
                        result
                        for result in results
                        if not is_reference_list_chunk(result.page_content)
                    ]

                    # Preserve retrieval ranking rather than grouping passages by page.
                    if not context_results:
                        st.warning(
                            "No usable passages were found after filtering reference entries."
                        )
                        st.stop()
                    context = "\n\n".join(
                        f"[Page {result.metadata['page']}]\n{result.page_content}"
                        for result in context_results
                    )

                    # Generate an answer using the retrived context.
                    valid_pages = {
                        result.metadata["page"] for result in context_results
                    }

                    with st.spinner(
                        "Generating an answer using retrieved evidence. "
                        "This may take a few minutes for complex questions..."
                    ):
                        start_time = time.perf_counter()

                        answer = generate_answer(question, context, valid_pages)

                        generation_time = time.perf_counter() - start_time

                    st.divider()
                    st.subheader("📊 Research Results")

                    with st.container(
                        border=True,
                        key="answer_panel",
                    ):
                        st.markdown("**Generated Answer**")
                        st.markdown(answer)
                    citation_check = validate_answer_citations(
                        answer,
                        valid_pages,
                    )

                    if not citation_check["has_citations"]:
                        st.warning(
                            "Citation validation: No PDF page citations were detected."
                        )

                    elif citation_check["invalid_pages"]:
                        st.warning(
                            "Citation validation: The answer cites "
                            "pages outside the retrieved context: "
                            f"{citation_check['invalid_pages']}"
                        )

                    else:
                        st.success(
                            "Citation validation passed. All detected "
                            "page citations refer to retrieved PDF pages."
                        )
                    st.markdown("**Query Performance**")

                    metric1, metric2, metric3 = st.columns(3)

                    metric1.metric(
                        "⚡ Retrieval Time",
                        f"{retrieval_time:.2f} s",
                    )

                    metric2.metric(
                        "⏱ Generation Time",
                        f"{generation_time:.2f} s",
                    )

                    metric3.metric(
                        "📄 Retrieved Passages",
                        len(results),
                    )
                    st.subheader("📑 Supporting Evidence")

                    st.caption(
                        "Explore the retrieved research passages "
                        "and their original PDF page references."
                    )

                    with st.expander(
                        f"View Retrieved Passages ({len(results)})",
                        expanded=False,
                    ):
                        for index, result in enumerate(results, start=1):
                            with st.container(border=True):
                                st.markdown(
                                    f"**Passage {index} | "
                                    f"PDF Page {result.metadata['page']}**"
                                )

                                st.write(result.page_content)

                else:
                    st.info("No relevant passages were found.")
            else:
                st.warning("Please enter a question.")

    except Exception as error:  # noqa: BLE001
        st.error(str(error))
# Application footer.
st.html(
    """
    <div class="workspace-footer">
        <p>COMP 699 · Professional Seminar Software · Rivier University</p>
    </div>
    """
)
