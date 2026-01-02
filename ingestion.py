import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load API Key from .env
load_dotenv()

# Configuration
DATA_PATH = "medical_data/"
DB_PATH = "chroma_db_medical"

def create_medical_db():
    # 1. Load PDFs from the folder
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📁 Created '{DATA_PATH}' folder. Please add your medical PDFs there.")
        return

    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    
    if len(raw_documents) == 0:
        print("⚠️ No PDFs found in 'medical_data/'. Please add some files first.")
        return

    # 2. Split text into chunks (Medical data needs precision)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    documents = text_splitter.split_documents(raw_documents)
    
    # 3. Create Embeddings & Store in ChromaDB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma.from_documents(
        documents, 
        embeddings, 
        persist_directory=DB_PATH
    )
    print(f"✅ Medical Database created successfully with {len(documents)} chunks.")

if __name__ == "__main__":
    create_medical_db()