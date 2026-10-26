from pypdf import PdfReader

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file."""
    reader = PdfReader(pdf_file)

    page_text = []
    for page in reader.pages:
        text=page.extract_text()
        page_text.append(text)
    
    full_text="\n\n".join(page_text)
    return len(reader.pages), full_text



    