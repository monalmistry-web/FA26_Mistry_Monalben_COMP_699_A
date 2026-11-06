from generation.answer_generator import generate_answer

def test_generate_answer():
    question = "What is retrieval-augmented generation?"

    context = """Retrieval-augmented generation combines information retrieval with language model generation. It retrieves relevant information and uses that information to generate an answer."""

    answer = generate_answer(question,context)

    assert answer is not None
    assert len(answer.strip())>0