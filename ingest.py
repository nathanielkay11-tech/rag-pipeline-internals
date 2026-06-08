import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env
load_dotenv()

# Load all PDFs from the test-data folder
print("Loading documents...")
loader = PyPDFDirectoryLoader("./test-data")
docs = loader.load()
print(f"Loaded {len(docs)} pages across all documents")

# Split into chunks
print("Chunking documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# Embed and store in Chroma
print("Embedding and storing in Chroma...")
embeddings = VoyageAIEmbeddings(
    voyage_api_key=os.getenv("VOYAGE_API_KEY"),
    model="voyage-law-2",
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

print("Done. Vector store written to ./chroma_db")