import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Load the existing Chroma vector store
embeddings = VoyageAIEmbeddings(
    voyage_api_key=os.getenv("VOYAGE_API_KEY"),
    model="voyage-law-2",
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

# Retrieve top 4 most relevant chunks for any question
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Claude Haiku — cheapest model, sufficient for Q&A against retrieved context
llm = ChatAnthropic(
    model="claude-haiku-4-5",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=512,
)

# Prompt — instructs Claude to answer only from the provided context
prompt = ChatPromptTemplate.from_template("""
You are a legal document assistant. Answer the question using only the context provided.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# The chain — retriever feeds into prompt feeds into Claude feeds into output parser
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    questions = [
        "What are the payment terms in the Accenture supply agreement?",
        "Who are the parties in the ASML litigation filing?",
        "What data protection obligations does the vendor have?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {rag_chain.invoke(q)}")