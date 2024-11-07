from flask import Flask, render_template, request, jsonify
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
import tempfile
import requests
from langchain_community.document_loaders import PyPDFLoader

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
    
    # Validate request contains required files/data
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if 'embeddings_model' not in request.form:
        return jsonify({'error': 'Embeddings Model URL is required'}), 400
        
    embeddings_model_url = request.form['embeddings_model'].strip()
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400
    
    # Validate URL format
    if not embeddings_model_url.startswith(('http://', 'https://')):
        return jsonify({'error': 'Invalid Embeddings Model URL format'}), 400
    
    # Initialize ChromaDB if not exists
    if chroma_db is None:
        if not init_chromadb(embeddings_model_url):
            return jsonify({'error': 'Failed to initialize ChromaDB. Please check the embeddings endpoint.'}), 500
    
    # Save uploaded file to temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            file.save(temp_file.name)
            
            # Verify file is not empty
            if os.path.getsize(temp_file.name) == 0:
                os.unlink(temp_file.name)
                return jsonify({'error': 'Uploaded file is empty'}), 400
            
            # Load and split PDF
            try:
                loader = PyPDFLoader(temp_file.name)
                pages = loader.load_and_split()
                
                if not pages:
                    return jsonify({'error': 'No content could be extracted from the PDF'}), 400
                
                # Extract text from pages and add to ChromaDB
                texts = [page.page_content for page in pages if page.page_content.strip()]
                
                if not texts:
                    return jsonify({'error': 'No valid text content found in PDF'}), 400
                
                chroma_db.add_texts(texts)
                
                return jsonify({
                    'message': f'PDF processed successfully. Added {len(texts)} pages to the database.',
                    'pages_processed': len(texts)
                })
                
            except Exception as e:
                return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Error handling file upload: {str(e)}'}), 500
        
    finally:
        # Ensure temporary file is always cleaned up
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass

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
