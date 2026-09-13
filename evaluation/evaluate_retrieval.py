import json
from pathlib import Path

import sys

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.append(str(SRC_DIR))

from hybrid_retrieval import hybrid_search


EVAL_SET_PATH = (
    BASE_DIR
    / "evaluation"
    / "retrieval_eval_set.json"
)


def load_eval_set():

    with open(
        EVAL_SET_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def get_rank(
    results,
    expected_chunk_id
):

    for rank, item in enumerate(
        results,
        start=1
    ):

        if item["chunk_id"] == expected_chunk_id:
            return rank

    return None


def hit_at_k(
    rank,
    k
):

    if rank is None:
        return 0

    return int(
        rank <= k
    )


def reciprocal_rank(
    rank
):

    if rank is None:
        return 0.0

    return 1.0 / rank


if __name__ == "__main__":

    eval_set = load_eval_set()

    dense_hit1 = []
    dense_hit3 = []
    dense_rr = []

    bm25_hit1 = []
    bm25_hit3 = []
    bm25_rr = []

    hybrid_hit1 = []
    hybrid_hit3 = []
    hybrid_rr = []


    for sample in eval_set:

        query = sample[
            "query"
        ]

        expected_chunk_id = sample[
            "expected_chunk_id"
        ]

        query_type = sample[
            "type"
        ]


        print("\n")
        print("=" * 90)

        print(
            "Query:",
            query
        )

        print(
            "Type:",
            query_type
        )

        print(
            "Expected chunk:",
            expected_chunk_id
        )


        # ====================================================
        # Out-of-domain 先跳过 Hit@K / MRR
        # ====================================================

        if expected_chunk_id is None:

            print(
                "这是 Out-of-domain Query，"
                "不参与 Hit@K / MRR。"
            )

            continue


        # ====================================================
        # Retrieval
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
        # Rank
        # ====================================================

        dense_rank = get_rank(
            dense_results,
            expected_chunk_id
        )

        bm25_rank = get_rank(
            bm25_results,
            expected_chunk_id
        )

        hybrid_rank = get_rank(
            hybrid_results,
            expected_chunk_id
        )


        print(
            "Dense Rank:",
            dense_rank
        )

        print(
            "BM25 Rank:",
            bm25_rank
        )

        print(
            "Hybrid Rank:",
            hybrid_rank
        )


        # ====================================================
        # Dense metrics
        # ====================================================

        dense_hit1.append(
            hit_at_k(
                dense_rank,
                1
            )
        )

        dense_hit3.append(
            hit_at_k(
                dense_rank,
                3
            )
        )

        dense_rr.append(
            reciprocal_rank(
                dense_rank
            )
        )


        # ====================================================
        # BM25 metrics
        # ====================================================

        bm25_hit1.append(
            hit_at_k(
                bm25_rank,
                1
            )
        )

        bm25_hit3.append(
            hit_at_k(
                bm25_rank,
                3
            )
        )

        bm25_rr.append(
            reciprocal_rank(
                bm25_rank
            )
        )


        # ====================================================
        # Hybrid metrics
        # ====================================================

        hybrid_hit1.append(
            hit_at_k(
                hybrid_rank,
                1
            )
        )

        hybrid_hit3.append(
            hit_at_k(
                hybrid_rank,
                3
            )
        )

        hybrid_rr.append(
            reciprocal_rank(
                hybrid_rank
            )
        )


    # ========================================================
    # Summary
    # ========================================================

    def average(values):

        if not values:
            return 0.0

        return (
            sum(values)
            / len(values)
        )


    print("\n")
    print("=" * 90)
    print("Retrieval Evaluation Summary")
    print("=" * 90)


    print("\nDense")

    print(
        "Hit@1:",
        round(
            average(dense_hit1),
            4
        )
    )

    print(
        "Hit@3:",
        round(
            average(dense_hit3),
            4
        )
    )

    print(
        "MRR:",
        round(
            average(dense_rr),
            4
        )
    )


    print("\nBM25")

    print(
        "Hit@1:",
        round(
            average(bm25_hit1),
            4
        )
    )

    print(
        "Hit@3:",
        round(
            average(bm25_hit3),
            4
        )
    )

    print(
        "MRR:",
        round(
            average(bm25_rr),
            4
        )
    )


    print("\nHybrid")

    print(
        "Hit@1:",
        round(
            average(hybrid_hit1),
            4
        )
    )

    print(
        "Hit@3:",
        round(
            average(hybrid_hit3),
            4
        )
    )

    print(
        "MRR:",
        round(
            average(hybrid_rr),
            4
        )
    )