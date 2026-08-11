def build_prompt(query, retrieved_results):

    # retrieval.py 返回的是：
    # {
    #     "chunk_id": ...,
    #     "distance": ...,
    #     "text": ...
    # }

    context_parts = []

    for i, result in enumerate(retrieved_results, start=1):

        context_parts.append(
            f"[参考资料 {i}]\n{result['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
你是一个企业内部知识库助手。

请严格根据下面提供的参考资料回答用户的问题。

要求：
1. 只能依据参考资料回答，不要编造不存在的信息。
2. 如果参考资料无法回答问题，请明确说明：
   “根据当前知识库，无法确定该问题的答案。”
3. 回答简洁、准确。
4. 如果资料只描述了规则，但没有给出具体数字，不要自行补充数字。
5. 必要时说明答案来自哪一条参考资料。

参考资料：

{context}


用户问题：

{query}


请给出回答：
""".strip()

    return prompt


if __name__ == "__main__":

    print("⑧ prompt_builder 开始处理检索结果", flush=True)

    from retrieval import search

    query = "公司的年假制度是什么？"

    retrieved_results = search(
        query=query,
        top_k=3
    )

    print("⑨ retrieval 结果已经返回", flush=True)
    print("retrieved_results:", retrieved_results, flush=True)

    prompt = build_prompt(
        query=query,
        retrieved_results=retrieved_results
    )

    print("⑩ Prompt 构造成功", flush=True)

    print("\n" + "=" * 60)
    print("最终 Prompt")
    print("=" * 60)
    print(prompt)