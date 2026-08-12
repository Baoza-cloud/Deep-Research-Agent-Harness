import json
import os
import sys


if __name__ == "__main__":

    # =========================
    # 1. 先检查参数
    # =========================

    if len(sys.argv) < 3:

        print(
            "retrieval_worker.py 是子进程文件，"
            "请通过 rag_pipeline.py 调用。"
        )

        sys.exit(1)


    # =========================
    # 2. 获取参数
    # =========================

    query = sys.argv[1]

    top_k = int(
        sys.argv[2]
    )


    # =========================
    # 3. 参数没问题之后
    # 才导入 Retrieval
    # =========================

    from retrieval import search


    # =========================
    # 4. 执行 Retrieval
    # =========================

    results = search(
        query=query,
        top_k=top_k
    )


    # =========================
    # 5. 返回结果
    # =========================

    result_json = json.dumps(
        results,
        ensure_ascii=False
    )

    sys.stdout.write(
        "__RETRIEVAL_RESULT__"
        + result_json
        + "\n"
    )

    sys.stdout.flush()


    # =========================
    # 6. 强制安全退出
    # =========================

    os._exit(0)