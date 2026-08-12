from pathlib import Path
import pickle

import numpy as np

from embedding_model import create_embedding


# =========================
# 1. 项目路径
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_PATH = BASE_DIR / "embedding" / "faiss.index"
CHUNKS_PATH = BASE_DIR / "embedding" / "chunks.pkl"


# =========================
# 2. Retrieval
# =========================

def search(query, top_k=3):

    # -------------------------
    # Step 1：先做 Query Embedding
    # -------------------------

    print("⑦ 进入 search()", flush=True)

    query_vector = create_embedding(
        [query]
    )

    query_vector = np.ascontiguousarray(
        query_vector,
        dtype=np.float32
    )

    print("⑧ query embedding 成功", flush=True)
    print(
        "query_vector shape:",
        query_vector.shape,
        flush=True
    )
    print(
        "query_vector dtype:",
        query_vector.dtype,
        flush=True
    )


    # -------------------------
    # Step 2：Embedding 完成后
    # 再加载 FAISS
    # -------------------------

    print("⑨ 准备加载 FAISS", flush=True)

    import faiss

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print("FAISS 加载成功", flush=True)


    # -------------------------
    # Step 3：加载 chunks
    # -------------------------

    with open(
        CHUNKS_PATH,
        "rb"
    ) as f:

        chunks = pickle.load(f)

    print(
        "FAISS dimension:",
        index.d,
        flush=True
    )

    print(
        "FAISS ntotal:",
        index.ntotal,
        flush=True
    )


    # -------------------------
    # Step 4：真正执行检索
    # -------------------------

    distances, indices = index.search(
        query_vector,
        top_k
    )

    print("⑩ FAISS search 成功", flush=True)


    # -------------------------
    # Step 5：根据 ID 找回 Chunk
    # -------------------------

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx == -1:
            continue

        if idx >= len(chunks):
            continue

        results.append({
            "chunk_id": int(idx),
            "distance": float(distance),
            "text": chunks[idx],

            # 今天新增加的 Metadata
            "source": "company_policy.txt"
        })

    return results


# =========================
# 3. 单独测试 Retrieval
# =========================

if __name__ == "__main__":

    query = "公司的年假制度是什么？"

    results = search(
        query=query,
        top_k=3
    )

    print("\n===== Retrieval Results =====")

    for result in results:

        print(
            "chunk_id:",
            result["chunk_id"]
        )

        print(
            "distance:",
            result["distance"]
        )

        print(
            "source:",
            result["source"]
        )

        print(
            "text:",
            result["text"]
        )

        print()