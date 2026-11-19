from generation.answer_generator import validate_answer_citations


def test_valid_page_citations():
    answer = "SCIG is reliable (Page 9). PMSG is efficient (Page 10)."

    result = validate_answer_citations(answer, {9, 10})

    assert result["has_citations"] is True
    assert result["invalid_pages"] == []
    assert result["is_valid"] is True

def test_invalid_page_citations():
    answer = "SCIG is reliable (Page 130). PMSG is efficient (Page 10)."

    result = validate_answer_citations(answer, {9, 10})

    assert result["has_citations"] is True
    assert result["invalid_pages"] == [130]
    assert result["is_valid"] is False


def test_missing_page_citations():
    answer = "SCIG is reliable and PMSG is efficient."

    result = validate_answer_citations(answer, {9, 10})

    assert result["has_citations"] is False
    assert result["invalid_pages"] == []
    assert result["is_valid"] is False

def test_valid_page_range():
    answer = "The paper discusses four generator types (Pages 9–10)."

    result = validate_answer_citations(answer, {9, 10})

    assert result["has_citations"] is True
    assert result["invalid_pages"] == []
    assert result["is_valid"] is True

def test_invalid_page_range():
    answer = "The paper discusses four generator types (Pages 9–12)."

    result = validate_answer_citations(answer, {9, 10})

    assert result["has_citations"] is True
    assert result["invalid_pages"] == [11, 12]
    assert result["is_valid"] is False