import os

from openai import OpenAI


# =========================
# 1. 读取 DeepSeek API Key
# =========================

API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise ValueError(
        "没有找到环境变量 DEEPSEEK_API_KEY"
    )


# =========================
# 2. 创建 DeepSeek Client
# =========================

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)


# =========================
# 3. 定义生成函数
# =========================

def generate_answer(prompt):

    print("① 开始调用 DeepSeek", flush=True)

    response = client.chat.completions.create(

        model="deepseek-v4-flash",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        # RAG问答不需要复杂推理，
        # 关闭thinking可以降低延迟和消耗
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },

        temperature=0.1,

        stream=False
    )

    print("② DeepSeek 返回成功", flush=True)

    answer = response.choices[0].message.content

    return answer


# =========================
# 4. 单独测试
# =========================

if __name__ == "__main__":

    test_prompt = """
请只回答一句话：
DeepSeek API 连接成功。
""".strip()

    answer = generate_answer(test_prompt)

    print("\n模型回答：")
    print(answer)