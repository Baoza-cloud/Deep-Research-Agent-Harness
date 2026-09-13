import json
from pathlib import Path
from hybrid_retrieval import hybrid_search
from relevance_filter import filter_results
from context_builder import build_context
from prompt_builder import build_prompt
from llm_client import generate_answer
from source_builder import build_sources


def rag_answer(
    question,
    dense_top_k=5,
    bm25_top_k=5,
    final_top_k=5,
    max_distance=1.0,
    max_chunks=3,
    rrf_k=60
):
    """
    V3 RAG Pipeline

    流程：

    1. Dense Retrieval
    2. BM25 Retrieval
    3. RRF Fusion
    4. Relevance Filter
    5. Context Builder
    6. Prompt Builder
    7. LLM Generate
    8. Source Builder
    """

    # ========================================================
    # 1. Hybrid Retrieval
    # ========================================================

    retrieval_result = hybrid_search(
        query=question,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        final_top_k=final_top_k,
        rrf_k=rrf_k
    )


    dense_results = retrieval_result[
        "dense_results"
    ]

    bm25_results = retrieval_result[
        "bm25_results"
    ]

    hybrid_results = retrieval_result[
        "hybrid_results"
    ]


    # ========================================================
    # 2. Relevance Filter
    # ========================================================

    filtered_results = filter_results(
        retrieved_results=hybrid_results,
        query=question,
        max_distance=max_distance,
        max_chunks=max_chunks
    )


    # ========================================================
    # 3. Fallback
    # ========================================================

    if not filtered_results:

        return {
            "question": question,

            "dense_results": dense_results,

            "bm25_results": bm25_results,

            "hybrid_results": hybrid_results,

            "filtered_results": [],

            "context": "",

            "prompt": "",

            "answer": (
                "根据当前企业知识库，"
                "没有检索到足够相关的资料，"
                "暂时无法回答该问题。"
            ),

            "sources": []
        }


    # ========================================================
    # 4. Build Context
    # ========================================================

    context = build_context(
        filtered_results
    )


    # ========================================================
    # 5. Build Prompt
    # ========================================================

    prompt = build_prompt(
        query=question,
        context=context
    )


    # ========================================================
    # 6. Generate Answer
    # ========================================================

    answer = generate_answer(
        prompt
    )


    # ========================================================
    # 7. Build Sources
    # ========================================================

    sources = build_sources(
        filtered_results
    )


    # ========================================================
    # 8. Return Structured Result
    # ========================================================

    return {
        "question": question,

        "dense_results": dense_results,

        "bm25_results": bm25_results,

        "hybrid_results": hybrid_results,

        "filtered_results": filtered_results,

        "context": context,

        "prompt": prompt,

        "answer": answer,

        "sources": sources
    }


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    test_questions = [
        "公司的年假制度是什么？",
        "工作满一年之后能休假吗？",
        "公司有没有健身房补贴？"
    ]


    all_results = []


    for question in test_questions:

        print("\n")
        print("=" * 90)
        print("Question:")
        print(question)
        print("=" * 90)


        result = rag_answer(
            question=question
        )


        # ====================================================
        # 保存本次结果
        # ====================================================

        all_results.append(
            result
        )


        # ====================================================
        # 打印 Answer
        # ====================================================

        print("\n")
        print("-" * 90)
        print("Answer")
        print("-" * 90)

        print(
            result["answer"]
        )


        # ====================================================
        # 打印 Sources
        # ====================================================

        print("\n")
        print("-" * 90)
        print("Sources")
        print("-" * 90)

        print(
            result["sources"]
        )


        # ====================================================
        # 打印 Retrieval Summary
        # ====================================================

        print("\n")
        print("-" * 90)
        print("Retrieval Summary")
        print("-" * 90)


        print(
            "Dense Results:",
            len(result["dense_results"])
        )

        print(
            "BM25 Results:",
            len(result["bm25_results"])
        )

        print(
            "Hybrid Results:",
            len(result["hybrid_results"])
        )

        print(
            "Filtered Results:",
            len(result["filtered_results"])
        )


        # ====================================================
        # 打印 Decision
        # ====================================================

        if len(
            result["filtered_results"]
        ) == 0:

            print(
                "Decision: FALLBACK"
            )

        else:

            print(
                "Decision: PASS TO LLM"
            )


            print("\n")
            print("-" * 90)
            print("Filtered Details")
            print("-" * 90)


            for rank, item in enumerate(
                result["filtered_results"],
                start=1
            ):

                print(
                    f"\nRank {rank}"
                )

                print(
                    "chunk_id:",
                    item["chunk_id"]
                )

                print(
                    "filter_reason:",
                    item.get(
                        "filter_reason"
                    )
                )

                print(
                    "dense_distance:",
                    item.get(
                        "dense_distance"
                    )
                )

                print(
                    "bm25_score:",
                    item.get(
                        "bm25_score"
                    )
                )

                print(
                    "rrf_score:",
                    item.get(
                        "rrf_score"
                    )
                )


    # ========================================================
    # 所有测试完成后，统一保存 JSON
    # ========================================================

    BASE_DIR = Path(
        __file__
    ).resolve().parent.parent


    evaluation_dir = (
        BASE_DIR
        / "evaluation"
    )


    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    output_path = (
        evaluation_dir
        / "baseline_v3_results.json"
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=4
        )


    print("\n")
    print("=" * 90)
    print("V3 Evaluation Results Saved")
    print("=" * 90)

    print(
        output_path
    )