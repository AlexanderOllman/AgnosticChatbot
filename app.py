from flask import Flask, render_template, request, jsonify
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
import tempfile
import requests

app = Flask(__name__)

# Global variables
CHROMADB_DIR = os.path.join(os.getcwd(), 'chromadb')
chroma_db = None

def init_chromadb(embeddings_url=None):
    """Initialize or load existing ChromaDB"""
    global chroma_db
    
    # Check if directory exists
    if not os.path.exists(CHROMADB_DIR):
        os.makedirs(CHROMADB_DIR)
    
    # If embeddings_url is provided, initialize with new embeddings
    if embeddings_url:
        try:
            models_url = f"{embeddings_url.rstrip('/')}/v1/models"
            response = requests.get(models_url)
            response.raise_for_status()
            models = response.json().get('data', [])
            model_name = models[0]['id'] if models else "custom-model"
            
            embeddings = OpenAIEmbeddings(
                model=model_name,
                openai_api_key="not-needed",
                openai_api_base=embeddings_url
            )
            chroma_db = Chroma(embedding_function=embeddings, persist_directory=CHROMADB_DIR)
            return True
        except Exception as e:
            print(f"Error initializing ChromaDB: {str(e)}")
            return False
    
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global chroma_db
    file = request.files['file']
    embeddings_model_url = request.form['embeddings_model']
    
    if not file or not embeddings_model_url:
        return jsonify({'error': 'File and Embeddings Model URL are required'}), 400
    
    # Initialize ChromaDB if not exists
    if chroma_db is None:
        if not init_chromadb(embeddings_model_url):
            return jsonify({'error': 'Failed to initialize ChromaDB'}), 500
    
    # Save uploaded file
    filepath = os.path.join(CHROMADB_DIR, file.filename)
    file.save(filepath)
    
    # Add documents
    with open(filepath, 'r') as f:
        content = f.read()
    chroma_db.add_texts([content])
    
    return jsonify({'message': 'File uploaded and processed successfully'})

@app.route('/chat', methods=['POST'])
def chat():
    if chroma_db is None:
        return jsonify({'error': 'No documents have been uploaded yet'}), 400
    
    user_message = request.json['message']
    llm_url = request.json['llm']
    
    if not llm_url or 'http' not in llm_url:
        return jsonify({'error': 'Valid LLM URL is required'}), 400
    
    # Get available models from the endpoint
    try:
        models_url = f"{llm_url.rstrip('/')}/v1/models"
        response = requests.get(models_url)
        response.raise_for_status()
        models = response.json().get('data', [])
        model_name = models[0]['id'] if models else "custom-model"
    except Exception as e:
        # Fallback to default if models endpoint fails
        model_name = "custom-model"
    
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key="not-needed",
        openai_api_base=llm_url
    )
    retriever = chroma_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    response = qa_chain.run(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    # Create ChromaDB directory if it doesn't exist
    if not os.path.exists(CHROMADB_DIR):
        os.makedirs(CHROMADB_DIR)
    app.run(debug=True, host='0.0.0.0', port='8080')
