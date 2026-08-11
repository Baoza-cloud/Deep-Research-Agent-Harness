import json
import os
import sys

from retrieval import search


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print(
            "retrieval_worker.py 是子进程文件，"
            "请运行 rag_pipeline.py。"
        )
        sys.exit(1)

    # 1. 接收参数
    query = sys.argv[1]
    top_k = int(sys.argv[2])

    # 2. 执行检索
    results = search(
        query=query,
        top_k=top_k
    )

    # 3. 转成 JSON
    result_json = json.dumps(
        results,
        ensure_ascii=False
    )

    # 4. 把结果写给主进程
    sys.stdout.write(
        "__RETRIEVAL_RESULT__"
        + result_json
        + "\n"
    )

    # 非常重要：确保结果已经真正输出
    sys.stdout.flush()

    # 5. 直接结束子进程
    # 不再让 Python 清理 FAISS / PyTorch 原生库
    os._exit(0)