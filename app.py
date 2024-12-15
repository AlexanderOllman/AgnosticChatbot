import os
import sqlite3
from flask import Flask, request, render_template, redirect, url_for, session
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure key in production.

DATABASE = 'conversations.db'

def init_db():
    if not os.path.exists(DATABASE):
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('''CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
        conn.close()

init_db()

def get_conversation():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM conversations ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return rows

def add_message(role, content):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("INSERT INTO conversations (role, content) VALUES (?,?)", (role, content))
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def chat():
    # Retrieve settings from session or use defaults
    base_url = session.get('llm_base_url', "https://llama-test-predictor-global-models.beta.hpepcai.com/v1")
    model = session.get('llm_model', "meta/llama3-8b-instruct")
    api_key = session.get('llm_api_key', "1234")
    vector_db_url = session.get('vector_db_url', "")

    if request.method == 'POST':
        user_message = request.form.get('message')
        use_vector_db = request.form.get('use_vector_db') == 'on'

        if user_message.strip():
            # Add user message to DB
            add_message('user', user_message)

            # If enabled, retrieve context from vector DB
            context = ""
            if use_vector_db and vector_db_url:
                # Perform retrieval
                retrieve_endpoint = f"{vector_db_url}/retrieve"
                params = {'query': user_message, 'top_k': 5}
                try:
                    r = requests.get(retrieve_endpoint, params=params)
                    if r.status_code == 200:
                        retrieved_docs = r.json().get('docs', [])
                        # Combine retrieved text into a context prompt
                        context = "\n\n".join([doc.get('content', '') for doc in retrieved_docs])
                except Exception as e:
                    print(f"Error during retrieval: {e}")

            # Construct final prompt
            prompt = user_message
            if context:
                prompt = f"Use the following context to answer:\n{context}\n\nUser: {user_message}"

            # Invoke the LLM
            llm = ChatNVIDIA(base_url=base_url, model=model, api_key=api_key)
            try:
                llm_response = llm.invoke(prompt)
            except Exception as e:
                llm_response = f"Error calling LLM: {e}"

            # Add assistant message to DB
            add_message('assistant', llm_response)

        return redirect(url_for('chat'))

    # Render chat UI
    conversation = get_conversation()
    return render_template('chat.html', conversation=conversation)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        llm_base_url = request.form.get('llm_base_url')
        llm_model = request.form.get('llm_model')
        llm_api_key = request.form.get('llm_api_key')
        vector_db_url = request.form.get('vector_db_url')

        session['llm_base_url'] = llm_base_url
        session['llm_model'] = llm_model
        session['llm_api_key'] = llm_api_key
        session['vector_db_url'] = vector_db_url

        return redirect(url_for('settings'))

    # Render settings form
    return render_template('settings.html',
                           llm_base_url=session.get('llm_base_url', ''),
                           llm_model=session.get('llm_model', ''),
                           llm_api_key=session.get('llm_api_key', ''),
                           vector_db_url=session.get('vector_db_url', ''))

@app.route('/upload', methods=['GET','POST'])
def upload():
    # This route remains as a simple single-file upload (if desired)
    if request.method == 'POST':
        file = request.files.get('file')
        vector_db_url = session.get('vector_db_url', "")
        if file and vector_db_url:
            upload_endpoint = f"{vector_db_url}/add"
            files = {'file': (file.filename, file.read())}
            try:
                r = requests.post(upload_endpoint, files=files)
                if r.status_code == 200:
                    return "File uploaded successfully!"
                else:
                    return f"Error uploading file: {r.text}", 400
            except Exception as e:
                return f"Exception during upload: {e}", 400
        else:
            return "No file or vector DB URL configured.", 400

    return '''
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file"/><br><br>
        <input type="submit" value="Upload to Vector DB"/>
    </form>
    '''

@app.route('/add_files', methods=['POST'])
def add_files():
    vector_db_url = session.get('vector_db_url', "")
    if not vector_db_url:
        return "Vector DB URL not configured.", 400

    # Get the list of uploaded files
    uploaded_files = request.files.getlist('files')
    if not uploaded_files:
        return "No files provided.", 400

    results = []
    for file in uploaded_files:
        if file.filename:
            upload_endpoint = f"{vector_db_url}/add"
            files = {'file': (file.filename, file.read())}
            try:
                r = requests.post(upload_endpoint, files=files)
                if r.status_code == 200:
                    results.append(f"{file.filename}: uploaded successfully")
                else:
                    results.append(f"{file.filename}: Error uploading file - {r.text}")
            except Exception as e:
                results.append(f"{file.filename}: Exception during upload - {e}")

    # After uploading, redirect back to settings
    session['upload_results'] = results
    return redirect(url_for('settings'))

@app.route('/clear_results', methods=['POST'])
def clear_results():
    session.pop('upload_results', None)
    return redirect(url_for('settings'))

if __name__ == '__main__':
    app.run(app, debug=True, host='0.0.0.0', port='8080')
