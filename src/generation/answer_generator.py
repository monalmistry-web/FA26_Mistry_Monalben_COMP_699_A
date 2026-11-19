from langchain_ollama import ChatOllama
from ollama import ResponseError
from httpx import ConnectError, TimeoutException
import re

def validate_answer_citations(answer, valid_pages):
    """Validate PDF page citations, including page ranges."""

    valid_pages = {int(page) for page in valid_pages}
    cited_pages = set()

    citation_patterns = re.findall(
        r"\bPages?\s+(\d+(?:\s*[-–]\s*\d+)?)",
        answer,
        flags=re.IGNORECASE,
    )

    for citation in citation_patterns:
        if re.search(r"[-–]", citation):
            start, end = map(
                int,
                re.split(r"\s*[-–]\s*", citation)
            )

            if start <= end and end - start <= 100:
                cited_pages.update(range(start, end + 1))
            else:
                cited_pages.update([start, end])
        else:
            cited_pages.add(int(citation))

    invalid_pages = cited_pages - valid_pages

    return {
        "has_citations": bool(cited_pages),
        "invalid_pages": sorted(invalid_pages),
        "is_valid": bool(cited_pages) and not invalid_pages,
    }

def generate_answer(question, context, valid_pages=None):
    """Generate an answer using retrieved research paper passages."""

    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )

    prompt = f"""
    You are an academic research paper assistant.

    Read ALL retrieved passages carefully before answering the question.
    Do not stop after finding a partial answer in one passage.

    Instructions:
    1. Examine every retrieved passage and identify information relevant to the question.
    2. Combine relevant evidence from multiple passages when necessary.
    3. Preserve the exact technical terms, categories, numbers, and relationships from the paper.
    4. Do not invent information, methods, categories, or page numbers.
    5. When the question asks for a list, check all passages for every relevant item before answering.
    6. Give a clear, concise explanation using only the retrieved evidence.
    7. Cite the actual supporting PDF page for each factual statement using (Page N).
    8. If the passages do not contain enough information, explain what is missing.

    Retrieved passages:
    Each [Page N] label identifies the actual PDF page number.
    Section numbers and reference numbers are not PDF page numbers.
    Use only the [Page N] labels when citing evidence.

    {context}

    Question:
    {question}

    Answer:
    """
    try:
        response = llm.invoke(prompt)

    except ConnectError as error:
        raise RuntimeError( "Ollama is not running. Please start Ollama and try again.")from error

    except TimeoutException as error:
        raise RuntimeError("Ollama took too long to respond. Please try again.")from error

    except ResponseError as error:
        if error.status_code == 404:
            raise RuntimeError("The required model 'llama3.2' is not installed. " "Please download it using 'ollama pull llama3.2'.")from error
        raise RuntimeError(f"Ollama returned an error: {error.error}") from error


    answer = response.content

    # Retry once if the generated answer has no PDF page citations.
    citations = re.findall(
        r"\bPages?\s+\d+\b",
        answer,
        flags=re.IGNORECASE,
    )

    if not citations:
        allowed_pages = sorted(
            {int(page) for page in valid_pages}
        ) if valid_pages is not None else []

        retry_prompt = f"""
        You previously answered this research question without
        including PDF page citations.

        Rewrite the answer using only the retrieved context below.

        Requirements:
        1. Keep the original answer's structure, but correct any unsupported or incorrect statements using the retrieved context.
        2. Verify every statement against the retrieved context.
        3. Include the supporting PDF page number for each factual statement.
        4. Every factual sentence MUST end with a citation in the format (Page N), where N is the PDF page containing its supporting evidence. Never return an answer without citations when supporting pages are available.
        5. The ONLY permitted PDF page numbers are: {allowed_pages}.
        6. These are PDF page numbers, NOT academic reference numbers.
        7. Never use numbers from references such as [130] as page citations.
        8. Match each citation to evidence on that specific PDF page.
        9. Do not invent page numbers or unsupported information.
        10. If a statement is unsupported, remove it.

        Context:
        {context}

        Question:
        {question}

        Previous answer:
        {answer}

        Revised answer:
        """

        try:
            retry_response = llm.invoke(retry_prompt)
            answer = retry_response.content

        except (ConnectError, TimeoutException, ResponseError):
            # Keep the original answer if the retry fails.
            pass

    return answer