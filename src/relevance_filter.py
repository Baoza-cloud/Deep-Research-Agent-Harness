import re
import jieba


# ============================================================
# 一些没有实际检索价值的常见词
# ============================================================

STOP_WORDS = {
    "的",
    "了",
    "吗",
    "呢",
    "啊",
    "是",
    "有",
    "没有",
    "有没有",
    "什么",
    "怎么",
    "如何",
    "多少",
    "公司",
    "员工",
    "规定",
    "制度"
}


# ============================================================
# 1. 提取 Query 中真正有检索价值的关键词
# ============================================================

def extract_query_keywords(query):

    tokens = jieba.lcut(
        query,
        cut_all=False
    )

    keywords = []

    for token in tokens:

        token = token.strip()

        # 空字符串不要
        if not token:
            continue

        # 停用词不要
        if token in STOP_WORDS:
            continue

        # 纯标点符号不要
        if re.fullmatch(
            r"[\W_]+",
            token
        ):
            continue

        keywords.append(token)

    return keywords


# ============================================================
# 2. 提取特殊编号 / 型号 / 英文实体
# ============================================================

def extract_special_terms(query):

    """
    例如：

    HR-017
    BX-2026-001
    XG-550
    GPT-4
    ABC123

    这种 token 对 BM25 很重要。
    """

    pattern = r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+"

    special_terms = re.findall(
        pattern,
        query
    )

    return special_terms


# ============================================================
# 3. 判断 BM25 是否具有“强关键词证据”
# ============================================================

def has_strong_bm25_evidence(
    query,
    result
):

    text = result["text"]

    bm25_score = result.get(
        "bm25_score"
    )

    # --------------------------------------------
    # 没有 BM25 分数，说明不是 BM25 召回
    # --------------------------------------------

    if bm25_score is None:
        return False


    # --------------------------------------------
    # BM25 <= 0，一般说明关键词匹配非常弱
    # --------------------------------------------

    if bm25_score <= 0:
        return False


    # ========================================================
    # 情况 1：
    # Query 中存在 HR-017 这种特殊编号
    # ========================================================

    special_terms = extract_special_terms(
        query
    )

    for term in special_terms:

        if term.lower() in text.lower():
            return True


    # ========================================================
    # 情况 2：
    # 普通中文关键词匹配
    # ========================================================

    query_keywords = extract_query_keywords(
        query
    )

    if len(query_keywords) == 0:
        return False


    matched_keywords = []

    for keyword in query_keywords:

        if keyword in text:
            matched_keywords.append(
                keyword
            )


    matched_count = len(
        matched_keywords
    )

    total_keywords = len(
        query_keywords
    )


    # ========================================================
    # 如果 Query 只有一个核心关键词
    #
    # 例如：
    #
    # 年假
    # 报销
    #
    # 精确出现即可认为 lexical evidence 较强
    # ========================================================

    if total_keywords == 1:

        return matched_count == 1


    # ========================================================
    # 如果 Query 有多个核心关键词
    #
    # 至少匹配两个
    # 并且匹配比例 >= 50%
    # ========================================================

    keyword_match_ratio = (
        matched_count
        / total_keywords
    )


    if (
        matched_count >= 2
        and keyword_match_ratio >= 0.5
    ):
        return True


    return False


# ============================================================
# 4. V3 Hybrid Relevance Filter
# ============================================================

def filter_results(
    retrieved_results,
    query,
    max_distance=1.0,
    max_chunks=3
):

    filtered_results = []


    for result in retrieved_results:

        keep_result = False

        filter_reason = None


        # ====================================================
        # 第一层：
        # Dense Semantic Evidence
        # ====================================================

        dense_distance = result.get(
            "dense_distance"
        )


        # --------------------------------------------
        # 兼容 V2 Dense Retrieval
        # --------------------------------------------

        if dense_distance is None:

            dense_distance = result.get(
                "distance"
            )


        # --------------------------------------------
        # Dense 足够相关
        # --------------------------------------------

        if (
            dense_distance is not None
            and dense_distance <= max_distance
        ):

            keep_result = True

            filter_reason = (
                "dense_semantic_match"
            )


        # ====================================================
        # 第二层：
        # BM25 Lexical Evidence
        # ====================================================

        elif has_strong_bm25_evidence(
            query=query,
            result=result
        ):

            keep_result = True

            filter_reason = (
                "bm25_lexical_match"
            )


        # ====================================================
        # 两层都失败
        # ====================================================

        if not keep_result:
            continue


        # ====================================================
        # 不直接修改原始 result
        # 复制一份，再添加调试信息
        # ====================================================

        filtered_result = dict(
            result
        )

        filtered_result[
            "filter_reason"
        ] = filter_reason


        filtered_results.append(
            filtered_result
        )


        # ====================================================
        # 最多给 LLM max_chunks 个 chunk
        # ====================================================

        if (
            len(filtered_results)
            >= max_chunks
        ):
            break


    return filtered_results

