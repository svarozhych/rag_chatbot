import os
import uuid
import time
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

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

#OpenTelemetry to Jaeger
resource = Resource.create({"service.name": "RAG-chatbot"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer("RAG-chatbot")

otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4318/v1/traces",
    headers={}
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

load_dotenv()

#Add caching
set_llm_cache(InMemoryCache())

#Set up OpenAI API and LangChain keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "RAG-Observability-Demo"

#Initialize components
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o-mini")
vector_store = InMemoryVectorStore(embeddings)

#Define session tracking variables
session_id = str(uuid.uuid4())
conversation_count = 0

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
    with tracer.start_as_current_span("document_retrieval") as span:
        start_time = time.time()
        retrieved_docs = vector_store.similarity_search(state["question"])
        retrieval_time = (time.time() - start_time) * 1000 

        span.set_attribute("service.name", "rag-chatbot")
        span.set_attribute("retrieval_count", len(retrieved_docs))
        span.set_attribute("query", state["question"])

        return {"context": retrieved_docs, "retrieval_time_ms": retrieval_time}


def generate(state: State):
    with tracer.start_as_current_span("answer_generation") as span:
        global conversation_count #To track conversation count
        conversation_count += 1 

        span.set_attribute("service.name", "rag-chatbot")
        span.set_attribute("session_id", session_id)
        span.set_attribute("conversation_number", conversation_count)

        docs_content = "\n\n".join(doc.page_content for doc in state["context"])

        #RAG prompt
        prompt = hub.pull("rlm/rag-prompt")
        messages = prompt.invoke(
            {"question": state["question"], "context": docs_content})
        response = llm.invoke(messages, config={
                                        "metadata": {
                                            "session_id": session_id,
                                            "conversation_number": conversation_count,
                                            "retrieval_count": len(state["context"]),
                                            "function_id": "rag_pipeline",
                                            "retrieval_time_ms": state.get("retrieval_time_ms", 0),
                                            }})
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
            print(f"Session: {session_id[:8]} | Conversation: {conversation_count}")

    except Exception as e:
        print(f"❌ Error: {e}")
