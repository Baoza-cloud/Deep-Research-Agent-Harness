import json
import sys
from pathlib import Path


# ============================================================
# 1. 路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"

EVAL_SET_PATH = BASE_DIR / "evaluation" / "retrieval_eval_set.json"


sys.path.append(str(SRC_DIR))


from rag_pipeline import rag_answer


# ============================================================
# 2. 加载评测集
# ============================================================


def load_eval_set():

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 3. 判断系统是否触发 Fallback
# ============================================================


def system_fallback(result):

    filtered_results = result.get("filtered_results", [])

    return len(filtered_results) == 0


# ============================================================
# 4. Main
# ============================================================

if __name__ == "__main__":
    eval_set = load_eval_set()

    correct_count = 0

    total_count = 0

    # 混淆矩阵四个量
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for sample in eval_set:
        query = sample["query"]

        expected_fallback = sample["should_fallback"]

        print("\n")
        print("=" * 90)

        print("Query:", query)

        print("Expected Fallback:", expected_fallback)

        # ====================================================
        # 运行完整 RAG Pipeline
        # ====================================================

        result = rag_answer(question=query)

        predicted_fallback = system_fallback(result)

        print("Predicted Fallback:", predicted_fallback)

        # ====================================================
        # 是否判断正确
        # ====================================================

        is_correct = predicted_fallback == expected_fallback

        print("Correct:", is_correct)

        total_count += 1

        if is_correct:
            correct_count += 1

        # ====================================================
        # 混淆矩阵
        #
        # Positive = 应该 Fallback
        # ====================================================

        if expected_fallback and predicted_fallback:
            true_positive += 1

        elif not expected_fallback and not predicted_fallback:
            true_negative += 1

        elif not expected_fallback and predicted_fallback:
            false_positive += 1

        elif expected_fallback and not predicted_fallback:
            false_negative += 1

    # ========================================================
    # 5. Accuracy
    # ========================================================

    if total_count == 0:
        fallback_accuracy = 0.0

    else:
        fallback_accuracy = correct_count / total_count

    # ========================================================
    # 6. 输出
    # ========================================================

    print("\n")
    print("=" * 90)

    print("Fallback Evaluation Summary")

    print("=" * 90)

    print("Total Samples:", total_count)

    print("Correct:", correct_count)

    print("Fallback Accuracy:", round(fallback_accuracy, 4))

    print("\nConfusion Matrix")

    print("True Positive:", true_positive)

    print("True Negative:", true_negative)

    print("False Positive:", false_positive)

    print("False Negative:", false_negative)
# ============================================================
# 7. 保存 Fallback Evaluation 结果
# ============================================================

output_result = {
    "total_samples": total_count,
    "correct_count": correct_count,
    "fallback_accuracy": fallback_accuracy,
    "confusion_matrix": {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    },
}


output_path = BASE_DIR / "evaluation" / "fallback_v3_results.json"


with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_result, f, ensure_ascii=False, indent=4)


print("\n")
print("=" * 90)
print("Fallback Evaluation Saved")
print("=" * 90)

print(output_path)
