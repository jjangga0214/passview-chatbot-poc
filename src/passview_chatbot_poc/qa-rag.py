# get_ipython().system('pip install unstructured')
# get_ipython().system('pip install sentence-transformers')
# get_ipython().system('pip install chromadb')
# get_ipython().system('pip install openai')

from langchain.document_loaders import TextLoader

documents = TextLoader("e:/data/AI.txt").load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

# 문서를 청크로 분할
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


# docs 변수에 분할 문서를 저장
docs = split_docs(documents)

from langchain.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Chromdb에 벡터 저장
from langchain.vectorstores import Chroma

db = Chroma.from_documents(docs, embeddings)

import os  # 운영체제(os) 모듈을 가져올 때 사용되는 라이브러리

os.environ["OPENAI_API_KEY"] = "sk-"  # openai 키 입력

from langchain.chat_models import ChatOpenAI

model_name = "gpt-3.5-turbo"  # gpt-3.5-turbo 모델 사용
llm = ChatOpenAI(model_name=model_name)

# Q&A 체인을 사용하여 쿼리에 대한 답변 얻기
from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# 쿼리를 작성하고 유사성 검색을 수행하여 답변을 생성,따라서 txt에 있는 내용을 질의해야 합니다
query = "AI란?"
matching_docs = db.similarity_search(query)
answer = chain.run(input_documents=matching_docs, question=query)
answer
