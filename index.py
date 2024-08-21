# -*- coding: utf-8 -*-
from langchain.document_loaders import TextLoader  # 单个txt文件
from langchain.document_loaders import DirectoryLoader  # 批量txt文件
from langchain_community.document_loaders import PyPDFLoader  # FDP文件
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter  # 文档拆分器
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS  # 向量数据库
from langchain_community.chat_models import ChatOllama  # Ollama加载大模型
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import transformers
from transformers import pipeline, AutoTokenizer
import torch
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate,MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import Ollama
# from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

from langchain import OpenAI, VectorDBQA
from langchain.chains.summarize import load_summarize_chain


# loader = TextLoader("./")  # 本地数据加载
# # loader = DirectoryLoader('./', glob='**/*.txt')  # 加载所有txt文件
# documents = loader.load()  # 将数据转成 document 对象，每个文件会作为一个 document
# documents = TextLoader("./", encoding='utf-8').load()
pdf_loader = PyPDFLoader('LLM.pdf', extract_images=True)   # 使用OCR解析pdf中图片里面的文字

# txt_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)  # 创建拆分器
# documents = txt_splitter.split_documents(documents)  # 文档分割
documents = pdf_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10))
print(f'documents:{len(documents)}')  # 输出拆分的数量

# llm = OpenAI(model_name="text-davinci-003", max_tokens=1500)  # 加载GPT-3模型，对拆分的文档进行总结


# 常见的向量数据库包括 Chroma、weaviate 和 FAISS
# 加载embedding模型，用于将documents向量化
embeddings = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base')  # "damo/nlp_corom_sentence-embedding_chinese-base-medical"
# embeddings = OpenAIEmbeddings()  # 初始化 openai 的 embeddings 对象
# docsearch = Chroma.from_documents(documents, embeddings)  # 计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
# docsearch = Chroma.from_documents(documents, embeddings, persist_directory="data/")  # 保存本地向量数据库，持久化数据
# docsearch.persist()
docsearchs = Chroma(persist_directory="data/", embedding_function=embeddings)  # 加载数据
# vector_db = FAISS.from_documents(documents, embeddings)  # 将documents插入到faiss本地向量数据库
# vector_db.save_local('LLM.faiss')
# vector_db = FAISS.load_local('LLM.faiss',embeddings)  # 加载faiss向量库，用于知识召回
# retriever = vector_db.as_retriever(search_kwargs={"k": 5})
retriever = docsearchs.as_retriever()  # 索引器，根据用户查询与嵌入式块之间的语义相似性获取附加上下文

# llm = ChatOllama(model='llama3.1:latest')  # 加载大模型
llm = ChatOllama(model='llama3.1')
# llm = Ollama(model="llama3.1")

# chain = load_summarize_chain(llm, chain_type="refine", verbose=True)  # 利用大模型对拆分的文档进行总结，创建总结链
# chain.run(documents[:5])  # 执行总结链，（为了快速演示，只总结前5段）

query = "大模型发展前景如何？"
# qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})  # 自定义 chain prompt，verbose=True#是否显示详细信息
qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)  # stuff问答系统类型，结合检索和生成的方法
# qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)

# result = qa_chain.invoke({"query": query})
# # chat_history.extend((HumanMessage(content=query), result))
# print(result)
# # chat_history=chat_history[-20:]  # 最新10轮对话

chat_history = []
while True:  # 开始对话
    query = input('query:')
    response = qa_chain.invoke({'query': query, 'chat_history': chat_history})
    chat_history.extend((HumanMessage(content=query), response))
    print(response)
    # chat_history = chat_history[-20:]  # 最新10轮对话


