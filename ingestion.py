import os
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

# initialize pinecone
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),  # find at app.pinecone.io
    environment=os.environ.get('PINECONE_ENV'),  # next to api key in console
)

def ingest_docs():
    loader = GoogleDriveLoader(
        folder_id="1LCYaEfZVxD6nfZrIi0uGBHlZ7acuJfx3",
        file_types=["document"],
        recursive=False
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, #chunk size cant be too small or will lose semantic meaning
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))
    Pinecone.from_documents(texts, embeddings, index_name='chatbot-googledoc-index') #this is the vectorstore



# if __name__ == '__main__':
#     ingest_docs()