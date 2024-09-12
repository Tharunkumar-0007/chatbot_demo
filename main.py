from flask import Flask, render_template, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import re
import torch
import logging
from logging.handlers import RotatingFileHandler
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/userdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
CORS(app)  # Enable CORS for all routes

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler = RotatingFileHandler('chatbot.log', maxBytes=1000000, backupCount=3)
file_handler.setFormatter(log_formatter)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# Define the User model matching the database table structure
class User(db.Model):
    __tablename__ = 'register'
    uname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), primary_key=True)
    password = db.Column(db.String(100), nullable=False)
    retype_password = db.Column(db.String(100), nullable=False)

# Global variables for the QA chain and other components
qa_chain = None
question_count = 0

# Simplified prompt template for faster generation
custom_prompt_template = """Answer the following question using the given context.
Context: {context}
Question: {question}
Helpful answer:
"""

def set_custom_prompt():
    """Prompt template for QA retrieval."""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    """Load the language model."""
    llm = CTransformers(
        model="TheBloke/llama-2-7b-chat-GGML",
        model_type="llama",
        max_new_tokens=256,
        temperature=0.9,
        n_gpu_layers=8,
        n_threads=24,
        n_batch=1000
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    """Create a RetrievalQA chain."""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    return qa_chain

def initialize_qa_bot():
    """Initialize the QA bot and store it in a global variable."""
    global qa_chain
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': device})
    
    try:
        db = FAISS.load_local("vectorstores/db_faiss", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        app.logger.error(f"Error loading FAISS database: {e}")
        return None
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa_chain = retrieval_qa_chain(llm, qa_prompt, db)

# Route to handle form submission for registration via AJAX
@app.route('/')
def home():
    return render_template('user.html')  # Ensure user.html exists in the 'templates' folder

@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    uname = data.get('uname')
    email = data.get('email')
    password = data.get('password')
    retype_password = data.get('retype_password')
    
    if not uname or not email or not password or not retype_password:
        app.logger.info(f'User registration attempt failed: {uname} - All fields are required.')
        return jsonify({'error': 'Please fill in all fields!'}), 400
    
    if password != retype_password:
        app.logger.info(f'User registration attempt failed: {uname} - Passwords do not match.')
        return jsonify({'error': 'Passwords do not match!'}), 400

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        app.logger.info(f'User registration attempt failed: {uname} - Email already registered.')
        return jsonify({'error': 'Email already registered!'}), 400
    
    new_user = User(uname=uname, email=email, password=password, retype_password=retype_password)
    db.session.add(new_user)
    db.session.commit()
    
    app.logger.info(f'User registered successfully: {uname}')
    return jsonify({'message': 'Registration successful!'}), 200

# Route to handle login
@app.route('/login', methods=['POST'])
def login_user():
    data = request.get_json()
    uname = data.get('uname')
    password = data.get('password')
    
    user = User.query.filter_by(uname=uname, password=password).first()
    
    if user:
        session['username'] = uname  # Store username in session
        app.logger.info(f'User logged in successfully: {uname}')
        return jsonify({'message': 'Login successful!'}), 200
    else:
        app.logger.info(f'Login attempt failed: {uname} - Invalid username or password.')
        return jsonify({'error': 'Invalid username or password!'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    global question_count
    data = request.get_json()  # Use get_json() to parse JSON request data
    user_input = data.get('query')
    username = session.get('username', 'Unknown')  # Retrieve username from session, default to 'Unknown'
    
    if not is_valid_query(user_input):
        app.logger.info(f'Question asked by {username}: {user_input}')
        return jsonify({"response": "Nothing matched. Please enter a valid query."})
    
    if qa_chain is None:
        app.logger.info(f'Question asked by {username}: {user_input}')
        return jsonify({"response": "Failed to initialize QA bot."})
    
    try:
        res = qa_chain({'query': user_input})
        answer = res.get("result", "No answer found.")
        question_count += 1
        app.logger.info(f'Question count: {question_count}')
        app.logger.info(f'Question asked by {username}: {user_input}')
        return jsonify({"response": answer, "username": username})
    except Exception as e:
        app.logger.error(f'Error processing the query by {username}: "{user_input}" - Error: {e}')
        return jsonify({"response": f"Error processing the query: {e}", "username": username})

def is_valid_query(query):
    """Check if the query is valid."""
    if not query or query.isspace():
        return False
    if not re.search(r'[a-zA-Z0-9]', query):
        return False
    return True

# Test route to check logging
@app.route('/test-logging')
def test_logging():
    app.logger.info("This is a test log entry.")
    return "Logging test complete."

if __name__ == '__main__':
    initialize_qa_bot()  # Initialize the QA bot when the app starts
    app.run(debug=True, port=5000)
