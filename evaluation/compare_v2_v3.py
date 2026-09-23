import json
from pathlib import Path


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

V2_PATH = BASE_DIR / "evaluation" / "baseline_v2_results.json"

V3_PATH = BASE_DIR / "evaluation" / "baseline_v3_results.json"


# ============================================================
# 2. 加载 JSON
# ============================================================


def load_json(path):

    if not path.exists():
        raise FileNotFoundError(f"没有找到文件：{path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 3. 判断是否触发 Fallback
# ============================================================


def is_fallback(result):

    filtered_results = result.get("filtered_results", [])

    return len(filtered_results) == 0


# ============================================================
# 4. 获取 V2 Retrieval Results
# ============================================================


def get_v2_retrieval(result):

    # V2 原来的字段叫 retrieval
    return result.get("retrieval", [])


# ============================================================
# 5. 获取 V3 Dense Results
# ============================================================


def get_v3_dense(result):

    return result.get("dense_results", [])


# ============================================================
# 6. 获取 V3 BM25 Results
# ============================================================


def get_v3_bm25(result):

    return result.get("bm25_results", [])


# ============================================================
# 7. 获取 V3 Hybrid Results
# ============================================================


def get_v3_hybrid(result):

    return result.get("hybrid_results", [])


# ============================================================
# 8. 获取 Top-1 chunk_id
# ============================================================


def get_top1_chunk_id(results):

    if not results:
        return None

    return results[0].get("chunk_id")


# ============================================================
# 9. 根据 question 建立索引
# ============================================================


def build_question_map(results):

    question_map = {}

    for result in results:
        question = result.get("question")

        if question is not None:
            question_map[question] = result

    return question_map


# ============================================================
# 10. 对比单个 Query
# ============================================================


def compare_query(question, v2_result, v3_result):

    v2_retrieval = get_v2_retrieval(v2_result)

    v3_dense = get_v3_dense(v3_result)

    v3_bm25 = get_v3_bm25(v3_result)

    v3_hybrid = get_v3_hybrid(v3_result)

    print("\n")
    print("=" * 100)

    print("Question:", question)

    print("=" * 100)

    # ========================================================
    # V2
    # ========================================================

    print("\n[V2]")

    print("Retrieval Count:", len(v2_retrieval))

    print("Filtered Count:", len(v2_result.get("filtered_results", [])))

    print("Top1 chunk_id:", get_top1_chunk_id(v2_retrieval))

    print("Fallback:", is_fallback(v2_result))

    print("Answer:", v2_result.get("answer", ""))

    # ========================================================
    # V3
    # ========================================================

    print("\n[V3]")

    print("Dense Count:", len(v3_dense))

    print("BM25 Count:", len(v3_bm25))

    print("Hybrid Count:", len(v3_hybrid))

    print("Filtered Count:", len(v3_result.get("filtered_results", [])))

    print("Dense Top1 chunk_id:", get_top1_chunk_id(v3_dense))

    print("BM25 Top1 chunk_id:", get_top1_chunk_id(v3_bm25))

    print("Hybrid Top1 chunk_id:", get_top1_chunk_id(v3_hybrid))

    print("Fallback:", is_fallback(v3_result))

    print("Answer:", v3_result.get("answer", ""))

    # ========================================================
    # V2 vs V3
    # ========================================================

    print("\n[Comparison]")

    v2_top1 = get_top1_chunk_id(v2_retrieval)

    v3_top1 = get_top1_chunk_id(v3_hybrid)

    print("Top1 Changed:", v2_top1 != v3_top1)

    print("V2 Top1:", v2_top1)

    print("V3 Hybrid Top1:", v3_top1)

    v2_fallback = is_fallback(v2_result)

    v3_fallback = is_fallback(v3_result)

    print("Fallback Changed:", v2_fallback != v3_fallback)

    print("V2 Fallback:", v2_fallback)

    print("V3 Fallback:", v3_fallback)


# ============================================================
# 11. Main
# ============================================================

if __name__ == "__main__":
    v2_results = load_json(V2_PATH)

    v3_results = load_json(V3_PATH)

    v2_map = build_question_map(v2_results)

    v3_map = build_question_map(v3_results)

    common_questions = [question for question in v2_map if question in v3_map]

    print("\n")
    print("=" * 100)

    print("V2 vs V3 Evaluation")

    print("=" * 100)

    print("V2 Samples:", len(v2_results))

    print("V3 Samples:", len(v3_results))

    print("Common Questions:", len(common_questions))

    for question in common_questions:
        compare_query(question=question, v2_result=v2_map[question], v3_result=v3_map[question])
