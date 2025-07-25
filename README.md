# RAG Chatbot with Unified Observability

A Retrieval-Augmented Generation (RAG) chatbot with comprehensive observability using **LangSmith** and **Jaeger** for multi-backend trace forwarding.

## 🎯 Overview

This project demonstrates how to add production-ready observability to an AI application by capturing detailed metadata and forwarding traces to multiple observability backends simultaneously.

## 🏗️ Architecture

```
User Question → PDF Retrieval → LLM Generation → Response
      ↓              ↓              ↓            ↓
   Metadata → Vector Search → OpenAI API → Answer
      ↓              ↓              ↓            ↓
  LangSmith ← Session Tracking ← Token Count ← Timing
      ↓
   Jaeger (via OpenTelemetry)
```

## 📊 Observability Features

### Metadata Captured
- **Session ID**: Unique identifier for user sessions
- **Conversation Number**: Sequential numbering within sessions  
- **Retrieval Metrics**: Document count, search timing
- **Token Usage**: Input/output tokens from OpenAI
- **Performance Timing**: Retrieval and generation latency
- **Function Tracking**: Operation identification

### Multi-Backend Forwarding
- **LangSmith**: Primary LLM observability platform
- **Jaeger**: Distributed tracing for infrastructure correlation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker (for Jaeger)
- OpenAI API key
- LangSmith API key

### Installation

1. **Clone and install dependencies:**
```bash
git clone <rag_chatbot>
cd rag-chatbot
pip install -r requirements.txt
```

2. **Set up environment variables:**
```bash
# Create .env file
OPENAI_API_KEY=your-openai-key-here
LANGCHAIN_API_KEY=your-langsmith-key-here
```

3. **Start Jaeger:**
```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 14268:14268 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

4. **Add your PDF document:**
```bash
# Place your PDF file in the project directory as "pa.pdf"
```

5. **Run the application:**
```bash
python rag_chatbot.py
```

## 🔍 Viewing Traces

### LangSmith Dashboard
1. Visit [https://smith.langchain.com](https://smith.langchain.com)
2. Navigate to project: "RAG-Observability-Demo"
3. View detailed LLM traces with business metadata

### Jaeger Dashboard  
1. Open [http://localhost:16686](http://localhost:16686)
2. Select service: "rag-chatbot"
3. View distributed traces with timing information

## 💡 How It Works

### 1. PDF Processing
- Loads PDF documents using PyPDFLoader
- Splits into 1000-character chunks with 200-character overlap
- Indexes in vector store for similarity search

### 2. RAG Pipeline
- **Retrieval**: Searches vector store for relevant chunks
- **Generation**: Uses OpenAI GPT-4o-mini with retrieved context
- **Response**: Returns generated answer with metadata

### 3. Observability Layer
- **LangSmith Integration**: Automatic LLM trace capture
- **OpenTelemetry Spans**: Manual instrumentation for custom metrics
- **Multi-backend**: Simultaneous forwarding to LangSmith + Jaeger

### 4. Metadata Flow
```python
# Session tracking
session_id = str(uuid.uuid4())
conversation_count += 1

# LangSmith metadata
config = {
    "metadata": {
        "session_id": session_id,
        "conversation_number": conversation_count,
        "retrieval_count": len(retrieved_docs),
        "function_id": "rag_pipeline"
    }
}

# Jaeger spans
with tracer.start_as_current_span("document_retrieval") as span:
    span.set_attribute("service.name", "rag-chatbot")
    span.set_attribute("query", question)
```

## 🎮 Example Usage

```bash
❓ Question: What is this document about?
🔍 Searching...
🤖 Answer: This document discusses ...

Timing: 45ms retrieval + 1200ms generation = 1245ms total
Tokens: 180 input + 65 output = 245 total
Model: gpt-4o-mini-2024-07-18 | Finish: stop
Session: 8bfaeed1 | Conversation: 3
```

## 📈 Business Value

### For Developers
- **Performance Debugging**: Identify slow retrieval or generation
- **Cost Optimization**: Track token usage patterns
- **Quality Monitoring**: Analyze retrieval relevance

### For Operations  
- **Resource Planning**: Predict scaling needs
- **Cost Management**: Monitor API expenses
- **Reliability**: Track system health and uptime

### For Product Teams
- **User Analytics**: Understand query patterns
- **Feature Usage**: Track functionality adoption
- **Business Intelligence**: Extract insights from interactions

## 🔒 Privacy & Security

- **Environment Variables**: API keys stored securely in .env
- **Session Isolation**: Each session independently tracked
- **Configurable Recording**: Control what content is captured
- **No PII Storage**: Only metadata and operational data collected

## 🎯 Key Learnings

1. **Multi-Backend Strategy**: Different tools serve different purposes
2. **Metadata Design**: Capture business-relevant metrics, not just technical
3. **OpenTelemetry Standard**: Enables vendor-neutral observability
4. **Real-World Challenges**: Integration complexity in enterprise environments

## 🔧 Troubleshooting

### LangSmith Not Working
- Verify `LANGCHAIN_API_KEY` in .env file
- Check project name in LangSmith dashboard
- Ensure network connectivity

### Jaeger Not Receiving Traces
- Verify Docker container is running: `docker ps`
- Check all ports are exposed: 16686, 14268, 4317, 4318
- Confirm OpenTelemetry setup is correct

### Performance Issues
- Monitor token usage patterns in LangSmith
- Optimize chunk size and overlap parameters
- Consider caching for repeated queries

## 📚 Additional Resources

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [OpenTelemetry Python Guide](https://opentelemetry-python.readthedocs.io/)
- [LangChain Observability](https://python.langchain.com/docs/guides/debugging)


**Built with ❤️ using LangChain, LangSmith, OpenTelemetry, and Jaeger**
