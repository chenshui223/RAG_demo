import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo',
    openai_api_base="https://api.chatanywhere.tech"
)

# response = chat.invoke("你好！")
# print(response)


from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Knock knock."),
    AIMessage(content="Who's there?"),
    HumanMessage(content="Orange"),
    
]

res = chat(messages)
# print(res)
messages.append(res)
res = chat(messages)
# print(res.content)
messages = [
    SystemMessage(content="你是一个专业的知识助手。"),
    HumanMessage(content="你知道baichuan2模型吗？"),
]
res = chat(messages)
print(res.content)

"""
很抱歉，我目前还没有关于"baichuan2模型"的相关信息。是否有其他问题我可以帮助您回答呢？
"""
