import json
import subprocess
import sys

from pathlib import Path

from prompt_builder import build_prompt
from llm_client import generate_answer
from context_builder import build_context
from prompt_builder import build_prompt
from source_builder import build_sources
from llm_client import generate_answer
from relevance_filter import filter_results

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

def rag_answer(
    question,
    top_k=5,
    max_distance=1.0
):

    # =========================
    # 1. Question
    # =========================

    print("\n① Question")
    print(question)


    # =========================
    # 2. Retrieval
    # =========================

    print("\n② Retrieval")

    retrieved_results = retrieve_in_subprocess(
        query=question,
        top_k=top_k
    )
    print("\n===== 原始 Retrieval 结果 =====")

    for result in retrieved_results:
        print(
            f"chunk_id={result['chunk_id']} | "
            f"distance={result['distance']:.4f} | "
            f"source={result['source']}"
        )
        print(result["text"])
        print()

    filtered_results = filter_results(
        retrieved_results,
        max_distance=1.0,
        max_chunks=3
    )

    print("\n===== Filter 后结果 =====")

    for result in filtered_results:
        print(
            f"chunk_id={result['chunk_id']} | "
            f"distance={result['distance']:.4f} | "
            f"source={result['source']}"
        )
        print(result["text"])
        print()

    # =========================
    # 3. Filter
    # =========================

    print("\n③ Relevance Filter")

    filtered_results = filter_results(
        retrieved_results,
        max_distance=max_distance,
        max_chunks=3
    )
    if not filtered_results:
        return {
            "question": question,
            "retrieval": retrieved_results,
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
    context = build_context(
        filtered_results
    )

    print("\n===== Context =====")
    print(context)

    # =========================
    # 4. Fallback
    # =========================

    if not filtered_results:

        answer = (
            "根据当前企业知识库，"
            "没有检索到足够相关的资料，"
            "暂时无法回答该问题。"
        )

        return {
            "question": question,
            "retrieval": retrieved_results,
            "filtered_results": [],
            "context": "",
            "prompt": "",
            "answer": answer,
            "sources": []
        }


    # =========================
    # 5. Context
    # =========================

    print("\n④ Context")

    context = build_context(
        filtered_results
    )


    # =========================
    # 6. Prompt
    # =========================

    print("\n⑤ Prompt")

    prompt = build_prompt(
        query=question,
        context=context
    )


    # =========================
    # 7. LLM
    # =========================

    print("\n⑥ LLM")

    answer = generate_answer(prompt)


    # =========================
    # 8. Sources
    # =========================

    sources = build_sources(
        filtered_results
    )


    # =========================
    # 9. Final Result
    # =========================

    return {
        "question": question,
        "retrieval": retrieved_results,
        "filtered_results": filtered_results,
        "context": context,
        "prompt": prompt,
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":

    import json
    from pathlib import Path

    # =========================
    # 1. 三个测试问题
    # =========================

    test_questions = [
        "公司的年假制度是什么？",
        "员工报销需要遵守什么规定？",
        "公司有没有健身房补贴？"
    ]


    # =========================
    # 2. 保存所有测试结果
    # =========================

    all_results = []


    # =========================
    # 3. 连续测试三个问题
    # =========================

    for i, question in enumerate(
        test_questions,
        start=1
    ):

        print("\n")
        print("=" * 70)
        print(f"测试问题 {i}")
        print("=" * 70)

        print("\nQuestion:")
        print(question)


        result = rag_answer(
            question=question,
            top_k=5,
            max_distance=1.0
        )


        # 保存结果
        all_results.append(result)


        # =====================
        # 打印 Retrieval
        # =====================

        print("\n----- 原始 Retrieval -----")

        for item in result["retrieval"]:

            print(
                f"chunk_id={item['chunk_id']} | "
                f"distance={item['distance']:.4f} | "
                f"source={item['source']}"
            )

            print(item["text"])
            print()


        # =====================
        # 打印 Filter 结果
        # =====================

        print("\n----- Filter 后 -----")

        if result["filtered_results"]:

            for item in result["filtered_results"]:

                print(
                    f"chunk_id={item['chunk_id']} | "
                    f"distance={item['distance']:.4f}"
                )

                print(item["text"])
                print()

        else:

            print("没有通过相关性筛选的资料")


        # =====================
        # 最终 Answer
        # =====================

        print("\n----- 最终回答 -----")
        print(result["answer"])


        # =====================
        # Sources
        # =====================

        print("\n----- Sources -----")

        if result["sources"]:

            for source in result["sources"]:
                print("-", source)

        else:
            print("无")


    # =========================
    # 4. 保存 JSON
    # =========================

    BASE_DIR = Path(__file__).resolve().parent.parent

    RESULT_PATH = (
        BASE_DIR
        / "evaluation"
        / "baseline_v2_results.json"
    )

    # 防止 evaluation 文件夹不存在
    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=2
        )


    print("\n")
    print("=" * 70)
    print("三个问题测试全部完成")
    print("=" * 70)

    print(
        "结果已保存到：",
        RESULT_PATH
    )



