from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import sys
import os
print(os.getenv("OPENAI_API_KEY"))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.data_loader import load_data

# Function to create and store embeddings
def create_vector_store(url):
    documents = load_data(url)
    if not documents:
        print("No documents loaded, cannot create vector store.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)

    vector_store.save_local("app/vectorstore/brainlox_faiss")
    print("Vector store created and saved successfully!")

if __name__ == "__main__":
    url = "https://brainlox.com/courses/category/technical"
    create_vector_store(url)


