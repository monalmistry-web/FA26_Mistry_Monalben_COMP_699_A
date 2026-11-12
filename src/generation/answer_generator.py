from langchain_ollama import ChatOllama

def generate_answer(question,context):
    """Generate an answer using retrieved research paper passages."""

    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )

    prompt = f"""
    You are an academic research paper assistant.

    Answer the user's question using only the information provided in the context below.

    Important instructions:
    1. Use only the information provided in the context to answer.
    2. Include the PDF page number for every factual statement.
    3. Use only the page numbers explicitly provided in the context.
    4. Match each factual statement to the page containing its supporting information.
    5. Do not use academic reference numbers such as [1], [2], or [3] as PDF page numbers.
    6. Do not invent information or page numbers.
    7. If the answer is not supported by the provided context, say that the information was not found in the research paper.
    8. Do not say that a page number is missing when it is provided in the context.
    9. If information comes from multiple pages, cite each relevant page.
    10. Answer the question directly and concisely. Do not include unrelated explanations or treat background information as part of the answer.
    11. When the context contains a direct answer to the question, prioritize that information over background explanations.
    12. Do not turn introductory statements, section headings, or background descriptions into numbered answers.
    13. When a question asks for a difference or comparison, explain both items and clearly state the key difference. Do not describe only one item.
    
    
    Context:
    {context}

    Question:
    {question}

    Answer:

    """
    response = llm.invoke(prompt)

    return response.content