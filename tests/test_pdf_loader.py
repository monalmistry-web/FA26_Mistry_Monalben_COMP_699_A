from io import BytesIO
from pypdf.errors import PdfReadError
from pypdf import PdfWriter
from unittest.mock import patch, MagicMock
import pytest


from ingestion.pdf_loader import extract_pages_from_pdf


def test_extract_pages_from_pdf():
    """Test PDF page count and empty-page handling."""

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)

    pdf_file = BytesIO()
    writer.write(pdf_file)
    pdf_file.seek(0)

    page_count, pages = extract_pages_from_pdf(pdf_file)

    assert page_count == 1
    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert pages[0]["text"] == ""


def test_extract_actual_text_from_pdf():
    """Test that text is extracted correctly from PDF pages."""

    fake_page = MagicMock()
    fake_page.extract_text.return_value = (
        "Introduction to Artificial Intelligence"
    )

    with patch("ingestion.pdf_loader.PdfReader") as mock_reader:
        mock_reader.return_value.is_encrypted = False
        mock_reader.return_value.pages = [fake_page]

        page_count, pages = extract_pages_from_pdf("sample.pdf")

    assert page_count == 1
    assert pages[0]["page"] == 1
    assert pages[0]["text"] == "Introduction to Artificial Intelligence"

def test_extract_pages_from_encrypted_pdf():
    """ Test that password-protected PDFs are rejected."""

    with patch("ingestion.pdf_loader.PdfReader") as mock_reader:
        mock_reader.return_value.is_encrypted = True

        with pytest.raises(ValueError, match="password-protected"):
            extract_pages_from_pdf("encrypted.pdf")

def test_extract_pages_from_invalid_pdf():
    """ Test that invalid PDF files produce a clear error."""

    with patch("ingestion.pdf_loader.PdfReader", side_effect=PdfReadError("Invalid PDF file")):
        with pytest.raises(ValueError, match="Unable to read the PDF"):
            extract_pages_from_pdf("invalid.pdf")
