# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pinecone
from backend.core import run_llm
import streamlit as st
from streamlit_chat import message
# Press the green button in the gutter to run the script.


st.header('Google Doc Chat Assistant - Anatomy Notes')

if 'user_prompt_history' not in st.session_state:
    st.session_state['user_prompt_history'] = [] #initialize to empty lists
if 'chat_answers_history' not in st.session_state:
    st.session_state['chat_answers_history'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

prompt = st.text_input("Prompt",placeholder="Enter your question here...")
if prompt:
    with st.spinner("Generating response..."):
        st.session_state['user_prompt_history'].append(prompt)
        generated_response = run_llm(prompt,st.session_state['chat_history'])#includes memory
        #st.write(generated_response['answer'])
        st.session_state['chat_answers_history'].append(generated_response['answer'])
        st.session_state['chat_history'].append((prompt,generated_response['answer']))

if st.session_state['chat_answers_history']:
    for generated_response,user_prompt in zip(st.session_state['chat_answers_history'],st.session_state['user_prompt_history']):
        message(user_prompt,is_user=True)
        message(generated_response)
# if __name__ == '__main__':
#     #ingest_docs()
#     #print(run_llm('What is the gall bladder?'))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
