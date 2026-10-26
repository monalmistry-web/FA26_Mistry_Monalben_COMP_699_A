import streamlit as st
from ingestion.pdf_loader import extract_text_from_pdf

st.set_page_config(page_title="Academic Research Paper Assistant",page_icon=":books:",layout="wide")

st.title("Academic Research Paper Assistant")
st.write("Upload your research paper in PDF format to extract text and analyze it.")

uploaded_file = st.file_uploader("Choose a research paper",type=["pdf"])

if uploaded_file is not None:
    try:
        page_count,text = extract_text_from_pdf(uploaded_file)
        st.success("PDF opened successfully!")
        st.write(f"**Number of pages:** {page_count}")

        if text.strip():
            st.text_area("Extracted Text", value=text, height=400)
        else:
            st.warning("No selectable text was found." "The PDF may contain scanned pages.")

    except Exception:
        st.error("Could not read this PDF. Please try a valid PDF file.")
