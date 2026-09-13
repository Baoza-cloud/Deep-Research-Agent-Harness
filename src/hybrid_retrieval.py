from bm25_retrieval import bm25_search
from pathlib import Path
import json
import subprocess
import sys
BASE_DIR = Path(__file__).resolve().parent.parent

RETRIEVAL_WORKER_PATH = (
    BASE_DIR
    / "src"
    / "retrieval_worker.py"
)
def dense_search_in_subprocess(
    query,
    top_k=5
):

    process = subprocess.run(
        [
            sys.executable,
            str(RETRIEVAL_WORKER_PATH),
            query,
            str(top_k)
        ],
        capture_output=True,
        text=True
    )

    stdout = process.stdout

    result_line = None

    for line in stdout.splitlines():

        if line.startswith(
            "__RETRIEVAL_RESULT__"
        ):
            result_line = line
            break


    if result_line is None:

        raise RuntimeError(
            "Dense Retrieval 没有返回有效结果。\n"
            f"退出代码：{process.returncode}\n"
            f"stdout：\n{process.stdout}\n"
            f"stderr：\n{process.stderr}"
        )


    result_json = result_line.replace(
        "__RETRIEVAL_RESULT__",
        "",
        1
    )

    dense_results = json.loads(
        result_json
    )

    return dense_results
def rrf_fusion(
    dense_results,
    bm25_results,
    top_k=5,
    rrf_k=60
):
    """
    使用 Reciprocal Rank Fusion (RRF)
    融合 Dense Retrieval 和 BM25 Retrieval 的结果。

    参数：
        dense_results:
            Dense Retrieval 返回的结果列表

        bm25_results:
            BM25 Retrieval 返回的结果列表

        top_k:
            最终返回前 K 个 Hybrid 结果

        rrf_k:
            RRF 平滑常数，默认 60

    返回：
        hybrid_results:
            融合并重新排序后的结果
    """

    fused_results = {}


    # ========================================================
    # 1. 处理 Dense Retrieval Results
    # ========================================================

    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        chunk_id = result["chunk_id"]

        # 如果这个 chunk 第一次出现
        if chunk_id not in fused_results:

            fused_results[chunk_id] = {
                "chunk_id": chunk_id,
                "text": result["text"],
                "source": result["source"],
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_distance": None,
                "bm25_score": None,
                "retrieval_type": "hybrid"
            }

        # 记录 Dense 排名
        fused_results[chunk_id]["dense_rank"] = rank

        # 保留原始 FAISS distance
        fused_results[chunk_id]["dense_distance"] = (
            result["distance"]
        )

        # 累加 Dense 的 RRF Score
        fused_results[chunk_id]["rrf_score"] += (
            1.0 / (rrf_k + rank)
        )


    # ========================================================
    # 2. 处理 BM25 Retrieval Results
    # ========================================================

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        chunk_id = result["chunk_id"]

        # 如果这个 chunk 没有出现在 Dense Results 中
        if chunk_id not in fused_results:

            fused_results[chunk_id] = {
                "chunk_id": chunk_id,
                "text": result["text"],
                "source": result["source"],
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_distance": None,
                "bm25_score": None,
                "retrieval_type": "hybrid"
            }

        # 记录 BM25 排名
        fused_results[chunk_id]["bm25_rank"] = rank

        # 保留原始 BM25 Score
        fused_results[chunk_id]["bm25_score"] = (
            result["bm25_score"]
        )

        # 累加 BM25 的 RRF Score
        fused_results[chunk_id]["rrf_score"] += (
            1.0 / (rrf_k + rank)
        )


    # ========================================================
    # 3. Dictionary → List
    # ========================================================

    hybrid_results = list(
        fused_results.values()
    )


    # ========================================================
    # 4. 根据 RRF Score 从高到低排序
    # ========================================================

    hybrid_results.sort(
        key=lambda result: result["rrf_score"],
        reverse=True
    )


    # ========================================================
    # 5. 只返回 Top-K
    # ========================================================

    return hybrid_results[:top_k]
def hybrid_search(
    query,
    dense_top_k=5,
    bm25_top_k=5,
    final_top_k=5,
    rrf_k=60
):

    print(
        "① 开始 Dense Retrieval",
        flush=True
    )

    dense_results = dense_search_in_subprocess(
        query=query,
        top_k=dense_top_k
    )

    print(
        f"② Dense Retrieval 完成，共 {len(dense_results)} 条",
        flush=True
    )


    print(
        "③ 开始 BM25 Retrieval",
        flush=True
    )

    bm25_results = bm25_search(
        query=query,
        top_k=bm25_top_k
    )

    print(
        f"④ BM25 Retrieval 完成，共 {len(bm25_results)} 条",
        flush=True
    )


    print(
        "⑤ 开始 RRF Fusion",
        flush=True
    )

    hybrid_results = rrf_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        top_k=final_top_k,
        rrf_k=rrf_k
    )

    print(
        "⑥ Hybrid Retrieval 完成",
        flush=True
    )

    return {
        "dense_results": dense_results,
        "bm25_results": bm25_results,
        "hybrid_results": hybrid_results
    }
from relevance_filter import filter_results


if __name__ == "__main__":

    test_queries = [
        "公司的年假制度是什么？",
        "工作满一年之后能休假吗？",
        "公司有没有健身房补贴？"
    ]


    for query in test_queries:

        print("\n")
        print("=" * 90)
        print("Query:", query)
        print("=" * 90)


        # ====================================================
        # 1. Hybrid Retrieval
        # ====================================================

        result = hybrid_search(
            query=query,
            dense_top_k=5,
            bm25_top_k=5,
            final_top_k=5,
            rrf_k=60
        )


        dense_results = result[
            "dense_results"
        ]

        bm25_results = result[
            "bm25_results"
        ]

        hybrid_results = result[
            "hybrid_results"
        ]


        # ====================================================
        # 2. 打印 Dense Retrieval 结果
        # ====================================================

        print("\n")
        print("-" * 90)
        print("Dense Retrieval Results")
        print("-" * 90)


        for rank, item in enumerate(
            dense_results,
            start=1
        ):

            print(
                f"\nDense Rank {rank}"
            )

            print(
                "chunk_id:",
                item["chunk_id"]
            )

            print(
                "distance:",
                round(
                    item["distance"],
                    4
                )
            )

            print(
                "source:",
                item["source"]
            )

            print(
                "text:",
                item["text"]
            )


        # ====================================================
        # 3. 打印 BM25 Retrieval 结果
        # ====================================================

        print("\n")
        print("-" * 90)
        print("BM25 Retrieval Results")
        print("-" * 90)


        for rank, item in enumerate(
            bm25_results,
            start=1
        ):

            print(
                f"\nBM25 Rank {rank}"
            )

            print(
                "chunk_id:",
                item["chunk_id"]
            )

            print(
                "bm25_score:",
                round(
                    item["bm25_score"],
                    4
                )
            )

            print(
                "source:",
                item["source"]
            )

            print(
                "text:",
                item["text"]
            )


        # ====================================================
        # 4. 打印 Hybrid RRF 结果
        # ====================================================

        print("\n")
        print("-" * 90)
        print("Hybrid Retrieval Results")
        print("-" * 90)


        for rank, item in enumerate(
            hybrid_results,
            start=1
        ):

            print(
                f"\nHybrid Rank {rank}"
            )

            print(
                "chunk_id:",
                item["chunk_id"]
            )

            print(
                "rrf_score:",
                round(
                    item["rrf_score"],
                    6
                )
            )

            print(
                "dense_rank:",
                item["dense_rank"]
            )

            print(
                "bm25_rank:",
                item["bm25_rank"]
            )

            print(
                "dense_distance:",
                item["dense_distance"]
            )

            print(
                "bm25_score:",
                item["bm25_score"]
            )

            print(
                "source:",
                item["source"]
            )

            print(
                "text:",
                item["text"]
            )


        # ====================================================
        # 5. V3 Relevance Filter
        # ====================================================

        filtered_results = filter_results(
            retrieved_results=hybrid_results,
            query=query,
            max_distance=1.0,
            max_chunks=3
        )


        # ====================================================
        # 6. 打印 Filter 后的结果
        # ====================================================

        print("\n")
        print("-" * 90)
        print("Filtered Hybrid Results")
        print("-" * 90)


        if not filtered_results:

            print(
                "\n没有通过 Relevance Filter 的结果。"
            )

            print(
                "后续 RAG Pipeline 应进入 Fallback。"
            )


        else:

            for rank, item in enumerate(
                filtered_results,
                start=1
            ):

                print(
                    f"\nFiltered Rank {rank}"
                )

                print(
                    "chunk_id:",
                    item["chunk_id"]
                )

                print(
                    "filter_reason:",
                    item["filter_reason"]
                )

                print(
                    "rrf_score:",
                    round(
                        item["rrf_score"],
                        6
                    )
                )

                print(
                    "dense_rank:",
                    item["dense_rank"]
                )

                print(
                    "bm25_rank:",
                    item["bm25_rank"]
                )

                print(
                    "dense_distance:",
                    item["dense_distance"]
                )

                print(
                    "bm25_score:",
                    item["bm25_score"]
                )

                print(
                    "source:",
                    item["source"]
                )

                print(
                    "text:",
                    item["text"]
                )


        # ====================================================
        # 7. 最终测试总结
        # ====================================================

        print("\n")
        print("-" * 90)
        print("Test Summary")
        print("-" * 90)


        print(
            "Dense Results:",
            len(dense_results)
        )

        print(
            "BM25 Results:",
            len(bm25_results)
        )

        print(
            "Hybrid Results:",
            len(hybrid_results)
        )

        print(
            "Filtered Results:",
            len(filtered_results)
        )


        if len(filtered_results) == 0:

            print(
                "Decision: FALLBACK"
            )

        else:

            print(
                "Decision: PASS TO LLM"
            )