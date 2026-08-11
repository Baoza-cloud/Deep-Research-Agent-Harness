import json
import subprocess
import sys

from pathlib import Path

from prompt_builder import build_prompt
from llm_client import generate_answer


# 当前 src 文件夹
SRC_DIR = Path(__file__).resolve().parent

# retrieval_worker.py 的路径
RETRIEVAL_WORKER = SRC_DIR / "retrieval_worker.py"


def retrieve_in_subprocess(query, top_k=3):

    print("① 启动独立 Retrieval 进程", flush=True)

    process = subprocess.run(
        [
            sys.executable,
            str(RETRIEVAL_WORKER),
            query,
            str(top_k)
        ],
        capture_output=True,
        text=True
    )

    # =========================
    # 先尝试读取检索结果
    # =========================

    for line in process.stdout.splitlines():

        if line.startswith("__RETRIEVAL_RESULT__"):

            json_text = line.replace(
                "__RETRIEVAL_RESULT__",
                "",
                1
            )

            results = json.loads(json_text)

            print(
                "② Retrieval 子进程完成",
                flush=True
            )

            return results


    # =========================
    # 没拿到结果，再判断是否失败
    # =========================

    if process.returncode != 0:

        print("Retrieval stdout:")
        print(process.stdout)

        print("Retrieval stderr:")
        print(process.stderr)

        raise RuntimeError(
            f"Retrieval 子进程运行失败，"
            f"退出代码：{process.returncode}"
        )


    raise RuntimeError(
        "Retrieval 子进程虽然结束，"
        "但没有找到检索结果"
    )

def rag_answer(query, top_k=3):

    # ============================
    # R：Retrieval
    # ============================

    retrieved_results = retrieve_in_subprocess(
        query=query,
        top_k=top_k
    )

    print("③ 检索完成", flush=True)


    # ============================
    # A：Augmentation
    # ============================

    print("④ 开始构造 Prompt", flush=True)

    prompt = build_prompt(
        query=query,
        retrieved_results=retrieved_results
    )

    print("⑤ Prompt 构造完成", flush=True)


    # ============================
    # G：Generation
    # ============================

    print("⑥ 开始调用 DeepSeek", flush=True)

    answer = generate_answer(prompt)

    print("⑦ DeepSeek 回答完成", flush=True)

    return answer


if __name__ == "__main__":

    query = "公司的年假制度是什么？"

    print("\n用户问题：")
    print(query)

    answer = rag_answer(
        query=query,
        top_k=3
    )

    print("\n" + "=" * 60)
    print("最终回答")
    print("=" * 60)

    print(answer)