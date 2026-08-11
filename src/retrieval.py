print("① retrieval.py 开始运行", flush=True)

from pathlib import Path
import pickle
import faiss
import numpy as np

print("② 基础库导入成功", flush=True)

from embedding_model import create_embedding

print("③ embedding_model 导入成功", flush=True)


BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_PATH = BASE_DIR / "embedding" / "faiss.index"
CHUNKS_PATH = BASE_DIR / "embedding" / "chunks.pkl"

print("④ 准备加载 FAISS", flush=True)

index = faiss.read_index(str(INDEX_PATH))

print("⑤ FAISS 加载成功", flush=True)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print("⑥ chunks 加载成功", flush=True)


# =========================
# 4. 定义检索函数
# =========================

def search(query, top_k=3):

    print("⑦ 进入 search()", flush=True)

    query_vector = create_embedding([query])

    print("⑧ query embedding 成功", flush=True)
    print("query_vector shape:", query_vector.shape, flush=True)
    print("query_vector dtype:", query_vector.dtype, flush=True)

    query_vector = np.ascontiguousarray(
        query_vector,
        dtype=np.float32
    )

    print("⑨ 准备执行 FAISS search", flush=True)
    print("FAISS dimension:", index.d, flush=True)
    print("FAISS ntotal:", index.ntotal, flush=True)

    distances, indices = index.search(
        query_vector,
        top_k
    )

    print("⑩ FAISS search 成功", flush=True)

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

    # 在 FAISS 中搜索 Top-K
    distances, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    # 根据 FAISS 返回的 ID 找回原始文本
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


# =========================
# 5. 测试
# =========================

if __name__ == "__main__":

    query = "公司的年假制度是什么？"

    results = search(
        query=query,
        top_k=3
    )

    print(f"\n用户问题：{query}\n")

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(f"===== Top {rank} =====")
        print("Chunk ID:", result["chunk_id"])
        print("L2 Distance:", result["distance"])
        print("Text:")
        print(result["text"])
        print()