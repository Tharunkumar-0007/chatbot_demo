from flask import Flask, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/userdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
CORS(app)  # Enable CORS for all routes

# Define the User model matching the database table structure
class User(db.Model):
    __tablename__ = 'register'  # Specify the table name
    uname = db.Column(db.String(100), nullable=False)  # Username field
    email = db.Column(db.String(100), primary_key=True)  # Email field is the primary key
    password = db.Column(db.String(100), nullable=False)  # Password field
    retype_password = db.Column(db.String(100), nullable=False)  # Retype password field

# Route to handle form submission for registration via AJAX
@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    uname = data.get('uname')
    email = data.get('email')
    password = data.get('password')
    retype_password = data.get('retype_password')
    
    # Check if all required fields are filled
    if not uname or not email or not password or not retype_password:
        return jsonify({'error': 'Please fill in all fields!'}), 400
    
    # Check if passwords match
    if password != retype_password:
        return jsonify({'error': 'Passwords do not match!'}), 400

    # Check if email already exists
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({'error': 'Email already registered!'}), 400
    
    # Add new user to the database
    new_user = User(uname=uname, email=email, password=password, retype_password=retype_password)
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({'message': 'Registration successful!'}), 200

# Route to handle login
@app.route('/login', methods=['POST'])
def login_user():
    data = request.get_json()
    uname = data.get('uname')
    password = data.get('password')
    
    # Check if the username and password match the database record
    user = User.query.filter_by(uname=uname, password=password).first()
    
    if user:
        # Successful login, redirect to user.html (This could be handled on the frontend)
        return jsonify({'message': 'Login successful! Redirecting to user page...'}), 200
    else:
        # Login failed
        return jsonify({'error': 'Invalid username or password!'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
