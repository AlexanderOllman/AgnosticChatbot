from flask import Flask, render_template, request, jsonify
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
import tempfile
import requests

app = Flask(__name__)

# In-memory ChromaDB
CHROMADB_DIR = tempfile.mkdtemp()
chroma_db = None

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
    
    # Get available models from the endpoint
    try:
        models_url = f"{embeddings_model_url.rstrip('/')}/v1/models"
        response = requests.get(models_url)
        response.raise_for_status()
        models = response.json().get('data', [])
        model_name = models[0]['id'] if models else "custom-model"
    except Exception as e:
        # Fallback to default if models endpoint fails
        model_name = "custom-model"
    
    # Initialize ChromaDB if not exists
    if chroma_db is None:
        embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key="not-needed",
            openai_api_base=embeddings_model_url
        )
        chroma_db = Chroma(embedding_function=embeddings, persist_directory=CHROMADB_DIR)
    
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
    app.run(debug=True, host='0.0.0.0', port='8080')
