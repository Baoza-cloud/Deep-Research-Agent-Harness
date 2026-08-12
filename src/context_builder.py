def build_context(filtered_results):

    context_parts = []

    for i, result in enumerate(
        filtered_results,
        start=1
    ):

        source = result["source"]
        text = result["text"]

        context_parts.append(
            f"[参考资料 {i}]\n"
            f"来源：{source}\n"
            f"内容：{text}"
        )

    context = "\n\n".join(context_parts)

    return context