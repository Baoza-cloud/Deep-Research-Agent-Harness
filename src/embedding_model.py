import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    cache_folder="../embedding/model",
)


def create_embedding(texts):

    embeddings = model.encode(
        texts
    )

    return embeddings.astype("float32")



if __name__ == "__main__":

    texts = [
        "员工请假需要提前提交申请",
        "公司提供员工福利制度"
    ]

    vectors = create_embedding(texts)


    print("向量数量:")
    print(len(vectors))


    print("第一个文本向量维度:")
    print(len(vectors[0]))