# 基于langchain创建简易的对话大模型
1. 领域精准问答
2. 数据更新频繁
3. 生成内容可解释可追溯
4. 数据隐私保护

基于`LangChain`, `OpenAI(LLM)`,  `vector DB`构建一个属于自己的LLM模型。

主要使用的技术————***Retrieval Augmented Generation (RAG)***
### 准备环境
``` 
! pip install -qU \
    langchain==0.0.316 \
    openai==0.28.1  \
    tiktoken==0.5.1  \
    cohere \
    chromadb==0.4.15 
    onnxruntime
    pypdf
    langchain-community
```

### 创建一个对话模型(no RAG)

#### 处理LLM存在的缺陷
1. 容易出现幻觉
2. 信息滞后
3. 专业领域深度知识匮乏


问题：你知道baichuan2模型吗？

回答：![alt text](/picture/p1.png)
### 创建一个RAG对话模型
#### 1. 加载数据 （以"https://arxiv.org/pdf/1706.03762"论文为例）
```
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("https://arxiv.org/pdf/1706.03762")

pages = loader.load_and_split()


```
#### 2. 知识切片 将文档分割成均匀的块。每个块是一段原始文本
```
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
)

docs = text_splitter.split_documents(pages)

```
#### 3. 利用embedding模型对每个文本片段进行向量化，并储存到向量数据库中
```
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


embed_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=docs, embedding=embed_model , collection_name="openai_embed")

```
#### 4. 通过向量相似度检索和问题最相关的K个文档。
```
query = "What dataset is used for training in the paper?"
result = vectorstore.similarity_search(query ,k = 2)
```
#### 5. 原始`query`与检索得到的文本组合起来输入到语言模型，得到最终的回答
```
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
  ```
### 创建一个对话模型(with RAG)
问题一：你知道baichuan2模型吗？

回答：![alt text](/picture/p2.png)

问题二：How large is the baichuan2 vocabulary?

回答：![alt text](/picture/p3.png)

### Note:
没有openai api的可以使用[GPT_API_free](https://github.com/chatanywhere/GPT_API_free)，很感谢他们对于开源项目的贡献。
