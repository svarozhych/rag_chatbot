import os
from typing import List, TypedDict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
from langgraph.graph import START, StateGraph
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from dotenv import load_dotenv

load_dotenv()

#Add caching
set_llm_cache(InMemoryCache())

#Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#Initialize components
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o-mini")
vector_store = InMemoryVectorStore(embeddings)

#Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def load_and_process_pdf(pdf_path: str):
    """Load and process PDF file into vector store"""
    print(f"Loading PDF: {pdf_path}")

    #Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    #Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    all_splits = text_splitter.split_documents(docs)

    #Index chunks
    _ = vector_store.add_documents(documents=all_splits)
    print(f"Processed {len(all_splits)} chunks from PDF")

    return all_splits

#Define application steps


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    #RAG prompt
    prompt = hub.pull("rlm/rag-prompt")
    messages = prompt.invoke(
        {"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


#Compile application
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

#Main execution
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found")
        exit(1)

    pdf_path = "pa.pdf"  #Add needed files here

    try:
        print("🤖 Setting up RAG system...")
        load_and_process_pdf(pdf_path)
        print("✅ Ready! Ask your questions.")
        print("Type 'quit' or 'exit' to stop.\n")

        while True:
            question = input("❓ Question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not question:
                continue

            print("🔍 Searching...")
            result = graph.invoke({"question": question})
            print(f"🤖 Answer: {result['answer']}\n")

    except Exception as e:
        print(f"❌ Error: {e}")
