from pathlib import Path
import pickle
import numpy as np

# 注意：
# 这里先不要 import faiss

from embedding_model import create_embedding


BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_PATH = BASE_DIR / "embedding" / "faiss.index"
CHUNKS_PATH = BASE_DIR / "embedding" / "chunks.pkl"


def search(query, top_k=3):

    # =============================
    # 1. 先生成 Query Embedding
    # =============================

    print("① 开始生成 query embedding", flush=True)

    query_vector = create_embedding([query])

    query_vector = np.ascontiguousarray(
        query_vector,
        dtype=np.float32
    )

    print("② query embedding 成功", flush=True)
    print("shape:", query_vector.shape, flush=True)
    print("dtype:", query_vector.dtype, flush=True)


    # =============================
    # 2. Embedding完成后再导入FAISS
    # =============================

    print("③ 开始加载 FAISS", flush=True)

    import faiss

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print("④ FAISS index 加载成功", flush=True)
    print("index.d:", index.d, flush=True)
    print("index.ntotal:", index.ntotal, flush=True)


    # =============================
    # 3. 加载 chunks
    # =============================

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print("⑤ chunks 加载成功", flush=True)


    # =============================
    # 4. 真正检索
    # =============================

    print("⑥ 开始 FAISS search", flush=True)

    distances, indices = index.search(
        query_vector,
        top_k
    )

    print("⑦ FAISS search 成功", flush=True)


    # =============================
    # 5. 根据 ID 找回文本
    # =============================

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx != -1 and idx < len(chunks):

            results.append({
                "chunk_id": int(idx),
                "distance": float(distance),
                "text": chunks[idx]
            })

    return results


if __name__ == "__main__":

    query = "公司的年假制度是什么？"

    results = search(
        query=query,
        top_k=3
    )

    print("\n============================")
    print("用户问题：", query)
    print("============================\n")

    for rank, result in enumerate(results, start=1):

        print(f"===== Top {rank} =====")
        print("Chunk ID:", result["chunk_id"])
        print("L2 Distance:", result["distance"])
        print("Text:")
        print(result["text"])
        print()