import os
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

from langchain.memory import ConversationBufferMemory


# initialize pinecone
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),  # find at app.pinecone.io
    environment=os.environ.get('PINECONE_ENV'),  # next to api key in console
)

def run_llm(query:str,chat_history):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))
    docsearch = Pinecone.from_existing_index(
                                        index_name='chatbot-googledoc-index',embedding=embeddings)  # this is the vectorstore

    chat = ChatOpenAI(verbose=True,temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(chat, docsearch.as_retriever(), memory=memory)
    #qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=docsearch.as_retriever(),return_source_documents=True)#as_retreiver function can turn a vector store into a retreiver
    return(qa({'question':query,'chat_history':chat_history}))
