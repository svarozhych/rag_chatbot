from flask import Flask, render_template, request, jsonify
import os
from rag_chatbot import graph, load_and_process_pdf, session_id, conversation_count

app = Flask(__name__)

# Initialize the RAG system on startup
def initialize_rag():
    pdf_path = "pa.pdf"  # Make sure this file exists
    if os.path.exists(pdf_path):
        print("🤖 Setting up RAG system...")
        load_and_process_pdf(pdf_path)
        print("✅ RAG system ready!")
    else:
        print(f"❌ PDF file '{pdf_path}' not found!")

# Initialize RAG system when the module is loaded
initialize_rag()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please enter a question'}), 400
        
        # Get answer from RAG system
        result = graph.invoke({"question": question})
        
        return jsonify({
            'answer': result['answer'],
            'session_id': session_id[:8],
            'conversation_count': conversation_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)