from pypdf import PdfReader

def extract_pages_from_pdf(pdf_file):
    """
    Extracts text from a PDF file."""
    reader = PdfReader(pdf_file)

    pages = []
    
    for page_number,page in enumerate(reader.pages,start=1):
        text = page.extract_text()
        pages.append({
            "page": page_number,
            "text": text or "",
        })
     
    return len(reader.pages), pages



    