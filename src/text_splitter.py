def split_text(text, chunk_size=100, overlap=50):
    """
    文本切分
    chunk_size:
        每个文本块大小
    overlap:
        两个chunk之间重叠长度
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

if __name__ == "__main__":

    text = """
    公司员工管理制度：
    第一章 总则，公司为了规范员工行为制定本制度。

    第二章 考勤管理：
    员工每天需要按时签到。
    如果迟到超过三次，将进行处罚。

    第三章 请假制度：
    员工请假需要提交申请。
    普通病假需要提前一天申请。

    第四章 年假制度：
    员工工作满一年后，可以享受带薪年假。
    年假天数根据员工工作年限计算。

    第五章 离职制度：
    员工离职需要提前提交申请。
    """

    result = split_text(text, chunk_size=100, overlap=20)

    for i, chunk in enumerate(result):

        print("================")
        print("Chunk", i)
        print(chunk)