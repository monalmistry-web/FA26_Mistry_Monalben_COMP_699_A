from pypdf import PdfReader
from pypdf.errors import PdfReadError

def extract_pages_from_pdf(pdf_file):
    """
    Extracts text from a PDF file."""
    try:
        reader = PdfReader(pdf_file)
        
        # Check whether the PDF is password-protected.
        if reader.is_encrypted:
            raise ValueError("This PDF is password-protected. Please upload an unprotected PDF.")

        # Check whether the PDF contains pages.
        if len(reader.pages) == 0:
            raise ValueError("The uploaded PDF contains no pages.")

        pages = []
    
        for page_number,page in enumerate(reader.pages,start=1):
            text = page.extract_text()
            pages.append({
            "page": page_number,
            "text": text or "",
            })
        return len(reader.pages), pages

    except ValueError:
        raise

    except (PdfReadError, OSError, EOFError) as error:
        raise ValueError(
        "Unable to read the PDF. Please upload a valid, uncorrupted PDF file."
        ) from error



    