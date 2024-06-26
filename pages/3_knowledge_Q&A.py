from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from llamaapi import LlamaAPI
import streamlit as st
import langchain as lc
from langchain import LLMMathChain
from langchain_openai import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_experimental.llms import ChatLlamaAPI
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from transformers import GPT2TokenizerFast
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document

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
        ('GPT-4o','GPT-3.5-turbo','Mixtral 8x7B', 'Llama-3-70B'))
    st.write('You selected:', option)

    st.write('You are using LangChain framework')

    # API Key input
    api_key = st.text_input("Please Copy & Paste your API_KEY", key="chatbot_api_key", type="password")

    # Reset button
    if st.button('Reset Conversation'):
        st.session_state["messages"] = []
        st.session_state["greeting_shown"] = False  # Reset the flag as well
        st.info("Please change your API_KEY if you change model.")

# Title and caption
st.title("💬 AI Chatbot")
st.caption("🚀 Your Personal AI Assistant powered by Streamlit and LLMs")

# Chat input
if not api_key:
    st.info("Please add your API_KEY to go ahead.")
    st.stop()

uploaded_file = st.file_uploader("Please upload a PDF", type="pdf")
if not uploaded_file:
    st.info("Please upload documents to continue.")
    st.stop()


if uploaded_file is not None:
    with open(uploaded_file.name, mode='wb') as w:
        w.write(uploaded_file.getvalue())
    loader = PyPDFLoader(uploaded_file.name)
    pdf_data = loader.load_and_split()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=250,
        length_function=count_tokens 
    )

    pdf_chunks = text_splitter.split_documents(pdf_data)
    st.write("PDF Splited by Chunks - You have {0} number of chunks.".format(len(pdf_data)))

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
    elif option=='GPT-3.5-turbo':
        chat = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', api_key=api_key)
        embeddings = OpenAIEmbeddings(api_key=api_key)
        db_FAISS = FAISS.from_documents(pdf_chunks, embeddings)
        retriever = db_FAISS.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6})
        qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, return_source_documents=True)
    elif option=='Mixtral 8x7B':
        chat = ChatMistralAI(temperature=0, model_name='open-mixtral-8x7b', api_key=api_key)
        embeddings = MistralAIEmbeddings(api_key=api_key)
        db_FAISS = FAISS.from_documents(pdf_chunks, embeddings)
        retriever = db_FAISS.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6})
        qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, return_source_documents=True)
    else:
        chat = ChatOpenAI(temperature=0, model_name='llama3-70b', api_key=api_key, base_url="https://api.llama-api.com")
        #llama = LlamaAPI(api_key=api_key,base_url="https://api.llama-api.com")
        #model = ChatLlamaAPI(client=llama, model_name='llama3-70b')
        embeddings = OpenAIEmbeddings(api_key=api_key,base_url="https://api.llama-api.com")
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

