import os
def load_text(file_path):
    """
    读取txt文件

    参数:
    file_path: 文件路径

    返回:
    文档文本内容
    """

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text



if __name__ == "__main__":

    file_path = "../data/raw/company_policy.txt"

    document = load_text(file_path)

    print("====文档内容====:")
    print(document)
