from openai import OpenAI
from mistralai import Mistral
from llamaapi import LlamaAPI
import streamlit as st
#from IPython.display import display, Math

# Sidebar for model selection
with st.sidebar:
    option = st.selectbox(
        'Please select your model',
        ('o1-mini','GPT-4o','GPT-4o-mini','GPT-4-turbo','GPT-3.5-turbo','Mixtral 8x7B','Mixtral 8x22B', 'Mistral Large 2','Mistral NeMo',
         'Llama-3.1-405B','Llama-3.2-3B','Llama-3.3-70B'))
    st.write('You selected:', option)

    # API Key input
    api_key = st.text_input("Please Copy & Paste your API_KEY", key="chatbot_api_key", type="password")

    # Reset button
    if st.button('Reset Conversation'):
        st.session_state["messages"] = []
        st.info("Please change your API_KEY if you change model.")
    
    

# Title and caption
st.title("💬 AI Chatbot")
st.caption("🚀 Your Personal AI Assistant powered by Streamlit and LLMs")



# Initialize messages if not present in session state
if 'messages' not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# Chat input
if prompt := st.chat_input():
    if not api_key:
        st.info("Please add your API_KEY to go ahead.")
        st.stop()

    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Client initialization based on selected model
    if option == 'Mixtral 8x7B':
        client = Mistral(api_key=api_key)
        response = client.chat.complete(model="open-mixtral-8x7b", messages=st.session_state.messages)
    elif option == 'Mixtral 8x22B':
        client = Mistral(api_key=api_key)
        response = client.chat.complete(model="open-mixtral-8x22b", messages=st.session_state.messages)
    elif option == 'Mistral Large 2':
        client = Mistral(api_key=api_key)
        response = client.chat.complete(model="mistral-large-2407", messages=st.session_state.messages)
    elif option == 'Mathstral':
        client = Mistral(api_key=api_key)
        response = client.chat.complete(model="mistralai/mathstral-7B-v0.1", messages=st.session_state.messages)
    elif option == 'Mistral NeMo':
        client = Mistral(api_key=api_key)
        response = client.chat.complete(model="open-mistral-nemo-2407", messages=st.session_state.messages)
    elif option == 'o1-mini':
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(model="o1-mini-2024-09-12", messages=st.session_state.messages)    
    elif option == 'GPT-4o':
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(model="gpt-4o", messages=st.session_state.messages) 
    elif option == 'GPT-4o-mini':
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(model="gpt-4o-mini", messages=st.session_state.messages)     
    elif option == 'GPT-4-turbo':
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(model="gpt-4-turbo", messages=st.session_state.messages)     
    elif option == 'GPT-3.5-turbo':
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    elif option == 'Llama-3.1-405B':
        client = OpenAI(api_key=api_key,base_url="https://api.llama-api.com")
        response = client.chat.completions.create(model="llama3.1-405b", messages=st.session_state.messages, max_tokens=1000)
    elif option == 'Llama-3.2-3B':
        client = OpenAI(api_key=api_key,base_url="https://api.llama-api.com")
        response = client.chat.completions.create(model="llama3.2-3b", messages=st.session_state.messages, max_tokens=1000) 
    elif option == 'Llama-3.3-70B':
        client = OpenAI(api_key=api_key,base_url="https://api.llama-api.com")
        response = client.chat.completions.create(model="llama3.3-70b", messages=st.session_state.messages, max_tokens=1000)        
    else:
        st.error("Selected model is not supported.")
        st.stop()

    def parse_stream_to_katex(stream: Stream):
        """
        Takes an OpenAI Stream and replaces ChatGPT LaTeX delimiters
        with KateX ones.
        Yields text, not chunks
        """
        last_text = ""
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                # Sometimes delimiters like \( can be split over two chunks.
                # If the previous chunk ended in \, prepend that to this chunk
                if last_text.endswith("\\"):
                    text = last_text + text

                text = (
                    text.replace(r"\[", "$$")
                    .replace(r"\]", "$$")
                    .replace(r"\(", "$")
                    .replace(r"\)", "$")
                )
                last_text = text

                # If the text ends in \, we don't return it, we'll get include it in the next chunk
                if not text.endswith("\\"):
                    yield text
                
    # Process response and update session state
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
