from openai import OpenAI
import streamlit as st
import langchain as lc
from langchain import LLMMathChain
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from transformers import GPT2TokenizerFast
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader



# Initialize the session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize the greeting_shown state
if "greeting_shown" not in st.session_state:
    st.session_state["greeting_shown"] = False

# Sidebar for model selection
with st.sidebar:
    option = st.selectbox(
        'Please select your model',
        ('GPT-4o','GPT-4o-mini','GPT-4-turbo','GPT-3.5-turbo'))
    st.write('You selected:', option)
with st.sidebar:
    doc = st.selectbox(
        'Please select your type of document',
        ('PDF','DOC','TXT'))
    st.write('You selected:', doc)



    st.write('You are using LangChain framework')

    # API Key input
    api_key = st.text_input("Please Copy & Paste your API_KEY", key="chatbot_api_key", type="password")

    # Reset button
    if st.button('Reset Conversation'):
        st.session_state["messages"] = []
        st.session_state["greeting_shown"] = False  # Reset the flag as well
        st.info("Please change your API_KEY if you change model.")

# Title and caption
st.title("💬 AI ChatDOC")
st.caption("🚀 Your Personal AI Assistant powered by Streamlit and LLMs")

# Chat input
if not api_key:
    st.info("Please add your API_KEY to go ahead.")
    st.stop()

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
if not uploaded_file:
    st.info("Please upload documents to continue.")
    st.stop()


if uploaded_file is not None:
    with open(uploaded_file.name, mode='wb') as w:
        w.write(uploaded_file.getvalue())
    if doc=='PDF':   
        loader = PyPDFLoader(uploaded_file.name)
        data = loader.load_and_split()
    elif doc=='DOC':
        loader = Docx2txtLoader(uploaded_file.name)
        data = loader.load()
    else:
        loader = TextLoader(uploaded_file.name)
        data = loader.load()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=250,
        length_function=count_tokens 
    )

    pdf_chunks = text_splitter.split_documents(data)
    st.write("Document Splited by Chunks - You have {0} number of chunks.".format(len(data)))

    # Add the greeting message only if it hasn't been shown yet
    if not st.session_state["greeting_shown"]:
        st.session_state["messages"].append({"role": "assistant", "content": "How can I help you?"})
        st.session_state["greeting_shown"] = True  # Set the flag to True

    
    if option=='GPT-4o':
        chat = ChatOpenAI(temperature=0, model_name='gpt-4o', api_key=api_key)
        embeddings = OpenAIEmbeddings(api_key=api_key)
        db_FAISS = FAISS.from_documents(pdf_chunks, embeddings)
        retriever = db_FAISS.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6})
        qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, return_source_documents=True)
    elif option=='GPT-4o-mini':
        chat = ChatOpenAI(temperature=0, model_name='gpt-4o-mini', api_key=api_key)
        embeddings = OpenAIEmbeddings(api_key=api_key)
        db_FAISS = FAISS.from_documents(pdf_chunks, embeddings)
        retriever = db_FAISS.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6})
        qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, return_source_documents=True)
    elif option=='GPT-3.5-turbo':
        chat = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', api_key=api_key)
        embeddings = OpenAIEmbeddings(api_key=api_key)
        db_FAISS = FAISS.from_documents(pdf_chunks, embeddings)
        retriever = db_FAISS.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6})
        qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, return_source_documents=True)
    else: 
        chat = ChatOpenAI(temperature=0, model_name='gpt-4-turbo', api_key=api_key)
        embeddings = OpenAIEmbeddings(api_key=api_key)
        db_FAISS = FAISS.from_documents(pdf_chunks, embeddings)
        retriever = db_FAISS.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6})
        qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, return_source_documents=True)

    
        


    
    if prompt := st.chat_input():
        st.session_state["messages"].append({"role": "user", "content": prompt})

 
        response = qa.invoke({"query": prompt})
        if "result" in response:
            st.session_state["messages"].append({"role": "assistant", "content": response["result"]})
        else:
            st.session_state["messages"].append({"role": "assistant", "content": "No 'result' key found in the response."})

    # Display the conversation
    for message in st.session_state["messages"]:
        if message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])
        elif message["role"] == "user":
            st.chat_message("user").write(message["content"])


