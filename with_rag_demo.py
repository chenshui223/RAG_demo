import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo',
    openai_api_base="https://api.chatanywhere.tech"
)

## 1.加载数据
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("https://arxiv.org/pdf/2309.10305.pdf")

pages = loader.load_and_split()


## 2.知识切片
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
)

docs = text_splitter.split_documents(pages)


"""
import requests
proxies = {"http": "http://127.0.0.1:10809"}  # 替换为你的代理
try:
    resp = requests.get("https://www.baidu.com", proxies=proxies, timeout=5)
    print("代理可用" if resp.status_code == 200 else "代理响应异常")
except Exception as e:
    print(f"代理不可用: {e}")
以上是测试网络代理，由于使用https://api.chatanywhere.tech，而不是默认的openai官网地址，所以需要另外设置代理
"""


## 3.使用embedding模型
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import httpx
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"
custom_proxies = {"http://":"http://127.0.0.1:10809",
                  "https://":"http://127.0.0.1:10809"}
embed_model = OpenAIEmbeddings(
    openai_api_base="https://api.chatanywhere.tech",  # 指定第三方服务地址
    http_client=httpx.Client(proxies=custom_proxies)  # 显式传递代理
)
vectorstore = Chroma.from_documents(documents=docs, embedding=embed_model , collection_name="openai_embed")

## 4.通过向量相似度检索和问题最相关K个内容
query = "How large is the baichuan2 vocabulary?"
result = vectorstore.similarity_search(query ,k = 2)
## 5.组合prompt
def augment_prompt(query: str):
  # 获取top3的文本片段
  results = vectorstore.similarity_search(query, k=3)
  source_knowledge = "\n".join([x.page_content for x in results])
  # 构建prompt
  augmented_prompt = f"""Using the contexts below, answer the query.

  contexts:
  {source_knowledge}

  query: {query}"""
  return augmented_prompt

# 创建prompt
prompt = HumanMessage(
    content=augment_prompt(query)
)

messages.append(prompt)

res = chat(messages)

print(res.content)
