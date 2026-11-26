import argparse
import hashlib
import json
import re
import sys
import time
from itertools import zip_longest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from generation.answer_generator import (
    generate_answer,
    validate_answer_citations,
)
from ingestion.pdf_loader import extract_pages_from_pdf
from ingestion.text_splitter import split_pages_into_chunks
from retrieval.vector_store import (
    add_chunks_to_vector_store,
    create_vector_store,
    expand_with_neighbor_chunks,
    search_hybrid_chunks,
    split_question_into_queries,
)

DATASET_PATH = ROOT_DIR / "evaluation" / "evaluation_dataset.json"
RESULTS_PATH = ROOT_DIR / "evaluation" / "evaluation_results.json"
PAPERS_DIR = ROOT_DIR / "data" / "papers"


def is_reference_list_chunk(text):
    """Apply the same bibliography filtering as app.py."""
    reference_entries = re.findall(
        r"\[\d{1,3}\]\s+[A-Z][A-Za-zÀ-ÿ.'’-]*\s+[A-Z]",
        text,
    )
    return len(reference_entries) >= 2


def prepare_paper(vector_store, pdf_name):
    """Reuse stored PDF chunks or process the PDF if needed."""
    pdf_path = PAPERS_DIR / pdf_name

    source_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    stored = vector_store.get(
        where={"source_id": source_id},
        include=[],
    )

    if not stored["ids"]:
        print(f"Processing PDF: {pdf_name}", flush=True)

        with pdf_path.open("rb") as pdf_file:
            _, pages = extract_pages_from_pdf(pdf_file)

        chunks = split_pages_into_chunks(pages)

        if not chunks:
            raise ValueError(f"No text chunks found: {pdf_name}")

        add_chunks_to_vector_store(vector_store, chunks, source_id)
    else:
        print(f"Using stored PDF: {pdf_name}", flush=True)

    return source_id


def evaluate_question(vector_store, item, source_id):
    """Execute the same Hybrid Search workflow as app.py."""
    question = item["question"]

    retrieval_start = time.perf_counter()

    queries = split_question_into_queries(question)

    results = []
    seen_chunks = set()
    query_results = []

    for query in queries:
        retrieved = search_hybrid_chunks(
            vector_store,
            query,
            source_id,
            k=4,
        )

        retrieved = [
            doc for doc in retrieved if not is_reference_list_chunk(doc.page_content)
        ]

        expanded = expand_with_neighbor_chunks(
            vector_store,
            retrieved,
            source_id,
            max_chunks=20,
        )

        query_results.append(expanded)

    retrieval_time = time.perf_counter() - retrieval_start

    for group in zip_longest(*query_results):
        for result in group:
            if result is None:
                continue

            chunk_key = (
                result.metadata.get("page"),
                result.page_content,
            )

            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                results.append(result)

            if len(results) >= 20:
                break

        if len(results) >= 20:
            break

    context_results = [
        result for result in results if not is_reference_list_chunk(result.page_content)
    ]

    if not context_results:
        raise ValueError("No usable passages were retrieved.")

    context = "\n\n".join(
        f"[Page {result.metadata['page']}]\n{result.page_content}"
        for result in context_results
    )

    valid_pages = {result.metadata["page"] for result in context_results}

    generation_start = time.perf_counter()

    answer = generate_answer(
        question,
        context,
        valid_pages,
    )

    generation_time = time.perf_counter() - generation_start

    citation_check = validate_answer_citations(
        answer,
        valid_pages,
    )

    retrieved_pages = sorted({result.metadata["page"] for result in results})

    expected_pages = set(item["expected_pages"])

    return {
        "id": item["id"],
        "pdf": item["pdf"],
        "question": question,
        "expected_answer": item["expected_answer"],
        "expected_pages": item["expected_pages"],
        "retrieved_pages": retrieved_pages,
        "context_pages": sorted(valid_pages),
        "matched_expected_pages": sorted(expected_pages.intersection(retrieved_pages)),
        "missing_expected_pages": sorted(expected_pages.difference(retrieved_pages)),
        "retrieved_passage_count": len(results),
        "retrieval_time_seconds": round(retrieval_time, 2),
        "generation_time_seconds": round(generation_time, 2),
        "generated_answer": answer,
        "citation_validation": citation_check,
        "retrieved_passages": [
            {
                "page": result.metadata["page"],
                "text": result.page_content,
            }
            for result in results
        ],
        "manual_assessment": "Pending",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Academic Research Paper Assistant."
    )

    parser.add_argument(
        "--question",
        help="Run only one question using its dataset ID.",
    )

    args = parser.parse_args()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    if args.question:
        dataset = [item for item in dataset if item["id"] == args.question]

        if not dataset:
            parser.error("Question ID was not found.")

    vector_store = create_vector_store()

    evaluation_results = []
    prepared_papers = {}

    output_path = RESULTS_PATH

    if args.question:
        output_path = (
            ROOT_DIR / "evaluation" / f"evaluation_results_{args.question}.json"
        )

    total_start = time.perf_counter()

    for index, item in enumerate(dataset, start=1):
        print(
            f"\n[{index}/{len(dataset)}] {item['id']}",
            flush=True,
        )

        pdf_name = item["pdf"]

        try:
            if pdf_name not in prepared_papers:
                prepared_papers[pdf_name] = prepare_paper(vector_store, pdf_name)

            result = evaluate_question(
                vector_store,
                item,
                prepared_papers[pdf_name],
            )

            print(
                f"Retrieved pages: {result['retrieved_pages']}",
                flush=True,
            )

            print(
                f"Generation time: {result['generation_time_seconds']} seconds",
                flush=True,
            )

            print(
                f"Citation validation: {result['citation_validation']['is_valid']}",
                flush=True,
            )

        except Exception as error:  # noqa: BLE001
            result = {
                **item,
                "error": str(error),
                "manual_assessment": "Pending",
            }

            print(f"Error: {error}", flush=True)

        evaluation_results.append(result)

        # Save after each question to preserve completed results.
        output_path.write_text(
            json.dumps(
                evaluation_results,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    total_time = time.perf_counter() - total_start

    print("\nEvaluation completed.")
    print(f"Total questions: {len(evaluation_results)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
