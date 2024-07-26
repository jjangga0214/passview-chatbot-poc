import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
import tempfile
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

uploaded_file = st.sidebar.file_uploader("upload", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    data = loader.load()

    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(data, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=30.0, model_name="gpt-4o"),
        retriever=vectors.as_retriever(),
    )

    def conversational_chat(query):  # 문맥 유지를 위해 과거 대화 저장 이력에 대한 처리
        result = chain({"question": query, "chat_history": st.session_state["history"]})
        st.session_state["history"].append((query, result["answer"]))
        return result["answer"]

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "generated" not in st.session_state:
        st.session_state["generated"] = [
            f"""안녕하세요! 저는 패스뷰 실험봇입니다.
            {uploaded_file.name} 을 잘 읽어보았습니다.
            저는 여러 실험을 진행할 수 있도록 디자인되었으므로, 
            사전에 특정한 명령을 받지 않은 상황입니다.
            따라서 실험하시고 싶은 작업을 "구체적으로" 말씀해주세요.
            저에게 특정한 역할을 부여해 주시고, 
            제가 어떻게 상호요용하면 좋을지 지정해주세요.
            """
        ]

    if "past" not in st.session_state:
        st.session_state["past"] = ["안녕하세요!"]

    # 챗봇 이력에 대한 컨테이너
    response_container = st.container()
    # 사용자가 입력한 문장에 대한 컨테이너
    container = st.container()

    with container:  # 대화 내용 저장(기억)
        with st.form(key="Conv_Question", clear_on_submit=True):
            user_input = st.text_input(
                "Query:",
                placeholder="패스뷰 관리자로서 실험하실 내용을 말씀해주세요. (:",
                key="input",
            )
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            output = conversational_chat(user_input)

            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with response_container:
            for i in range(len(st.session_state["generated"])):
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=str(i) + "_user",
                    avatar_style="fun-emoji",
                    seed="Nala",
                )
                message(
                    st.session_state["generated"][i],
                    key=str(i),
                    avatar_style="bottts",
                    seed="Fluffy",
                )
