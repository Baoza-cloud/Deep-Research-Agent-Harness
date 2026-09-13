from pathlib import Path
import pickle

import jieba
import numpy as np
from rank_bm25 import BM25Okapi


# ============================================================
# 1. 项目路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = (
    BASE_DIR
    / "embedding"
    / "chunks.pkl"
)


# ============================================================
# 2. 中文分词
# ============================================================

def tokenize_text(text):
    """
    对中文文本进行分词。

    参数：
        text: 原始文本字符串

    返回：
        tokens: 分词后的 token 列表
    """

    tokens = jieba.lcut(
        text,
        cut_all=False
    )

    # 去掉空字符串、空格等无效 token
    tokens = [
        token.strip()
        for token in tokens
        if token.strip()
    ]

    return tokens


# ============================================================
# 3. BM25 Retrieval
# ============================================================

def bm25_search(
    query,
    top_k=5
):
    """
    使用 BM25 对知识库 chunks 进行关键词检索。

    参数：
        query:
            用户问题

        top_k:
            返回最相关的前 K 个 chunk

    返回：
        results:
            BM25 检索结果列表
    """

    print("① 开始 BM25 Retrieval", flush=True)


    # --------------------------------------------------------
    # Step 1：加载 chunks
    # --------------------------------------------------------

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"没有找到 chunks 文件：{CHUNKS_PATH}"
        )

    with open(
        CHUNKS_PATH,
        "rb"
    ) as f:

        chunks = pickle.load(f)

    print(
        f"② chunks 加载成功，共 {len(chunks)} 个",
        flush=True
    )


    # 防止知识库为空
    if len(chunks) == 0:
        return []


    # --------------------------------------------------------
    # Step 2：对整个知识库进行中文分词
    # --------------------------------------------------------

    tokenized_corpus = [
        tokenize_text(chunk)
        for chunk in chunks
    ]

    print(
        "③ 文档分词完成",
        flush=True
    )


    # --------------------------------------------------------
    # Step 3：建立 BM25
    # --------------------------------------------------------

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    print(
        "④ BM25 索引建立完成",
        flush=True
    )


    # --------------------------------------------------------
    # Step 4：对 Query 进行分词
    # --------------------------------------------------------

    tokenized_query = tokenize_text(
        query
    )

    print(
        "⑤ Query 分词结果：",
        tokenized_query,
        flush=True
    )


    # --------------------------------------------------------
    # Step 5：计算每个 Chunk 的 BM25 Score
    # --------------------------------------------------------

    scores = bm25.get_scores(
        tokenized_query
    )

    print(
        "⑥ BM25 打分完成",
        flush=True
    )


    # --------------------------------------------------------
    # Step 6：从高到低排序
    # --------------------------------------------------------

    # top_k 不能超过 chunks 数量
    actual_top_k = min(
        top_k,
        len(chunks)
    )

    top_indices = np.argsort(
        scores
    )[::-1][:actual_top_k]


    # --------------------------------------------------------
    # Step 7：构造统一的 Retrieval Results
    # --------------------------------------------------------

    results = []

    for idx in top_indices:

        result = {
            "chunk_id": int(idx),

            # BM25 分数：越大代表关键词相关性通常越强
            "bm25_score": float(
                scores[idx]
            ),

            "text": chunks[idx],

            # 当前知识库只有一个主要文件，
            # 所以第一版先直接使用这个 source
            "source": "company_policy.txt",

            # 标记结果来自 BM25
            "retrieval_type": "bm25"
        }

        results.append(result)


    print(
        "⑦ BM25 Retrieval 完成",
        flush=True
    )

    return results


# ============================================================
# 4. 单独测试 BM25
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Test Query
    # --------------------------------------------------------

    query = "公司的年假制度是什么？"


    # --------------------------------------------------------
    # BM25 Search
    # --------------------------------------------------------

    results = bm25_search(
        query=query,
        top_k=5
    )


    # --------------------------------------------------------
    # 打印结果
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "BM25 Retrieval Results"
    )

    print("=" * 70)


    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            "chunk_id:",
            result["chunk_id"]
        )

        print(
            "bm25_score:",
            round(
                result["bm25_score"],
                4
            )
        )

        print(
            "source:",
            result["source"]
        )

        print(
            "retrieval_type:",
            result["retrieval_type"]
        )

        print(
            "text:"
        )

        print(
            result["text"]
        )

        print(
            "-" * 70
        )