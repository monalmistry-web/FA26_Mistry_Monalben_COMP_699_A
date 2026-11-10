from langchain_ollama import ChatOllama

def generate_answer(question,context):
    """Generate an answer using retrieved reasearch paper passages."""

    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )

    prompt = f"""
    You are an academic reasearch paper assistant.

    Answer the user's question using only the information provided in the context below.

    If the answer is not avilable in the context,
    say that the information was not found in the reasearch papaer.

    Context:
    {context}

    Question:
    {question}

    Answer:

    """
    response = llm.invoke(prompt)

    return response.content