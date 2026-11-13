import streamlit as st
import hashlib

from ingestion.pdf_loader import extract_pages_from_pdf
from ingestion.text_splitter import split_pages_into_chunks
from retrieval.vector_store import (create_vector_store,add_chunks_to_vector_store,search_similar_chunks)
from generation.answer_generator import generate_answer

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

        if st.button("Search"):
            if question.strip():
                with st.spinner("Searching for relevant passages..."):
                    vector_store = create_vector_store()
                    results=search_similar_chunks(
                        vector_store,
                        question,
                        source_id,
                        k=5)

                if results:
                    # Display the retrived passages.
                    for index, result in enumerate(results, start=1):
                        st.write(f"**Result {index} - Page {result.metadata['page']}**")

                        st.write(result.page_content)
                    # Prepare context for the language model.
                    context = "\n\n".join(
                        f"[Page {result.metadata['page']}\n{result.page_content}]"
                        for result in results
                    )

                    # Generate an answer using the retrived context.
                    with st.spinner("Generating answer..."):
                        answer = generate_answer(question,context)

                    st.subheader("Generated Answer")
                    st.write(answer)
                    
                else:
                    st.info("No relevant passages were found.")
            else:
                st.warning("Please enter a question.")

        
    except Exception as error:
        st.error(str(error))
