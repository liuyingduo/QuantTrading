from zhipuai import ZhipuAI
client = ZhipuAI(api_key="bb4a7915bf89f0eebf0f241c12c781d8.z7A8yZWbGDO1Wyjd")  # 请填写您自己的APIKey


def zhipu_api(messages):
    response = client.chat.completions.create(
        model="glm-4v-plus",  # 请填写您要调用的模型名称
        messages=messages,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "你好"},
    ]
    response = zhipu_api(messages)
    print(response)