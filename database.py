import sqlite3
import hashlib
import os
import json

def get_db_connection():
    """Create a connection to the SQLite database"""
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """Initialize the database with tables if they don't exist"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create user_profiles table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        profile_data TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    
def hash_password(password):
    """Hash a password for secure storage"""
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt + key

def verify_password(stored_password, provided_password):
    """Verify a stored password against a provided password"""
    salt = stored_password[:32]
    stored_key = stored_password[32:]
    key = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
    return key == stored_key

def register_user(username, password, email):
    """Register a new user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            conn.close()
            return False, "Username already exists"
        
        # Check if email already exists
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return False, "Email already exists"
        
        # Hash the password
        password_hash = hash_password(password)
        
        # Insert the new user
        cursor.execute(
            'INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)',
            (username, password_hash, email)
        )
        
        conn.commit()
        conn.close()
        return True, "User registered successfully"
    except Exception as e:
        return False, f"Registration error: {str(e)}"

def authenticate_user(username, password):
    """Authenticate a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return False, "Invalid username or password"
        
        stored_password = user['password_hash']
        if verify_password(stored_password, password):
            conn.close()
            return True, user['id']
        else:
            conn.close()
            return False, "Invalid username or password"
    except Exception as e:
        return False, f"Authentication error: {str(e)}"

def save_user_profile(user_id, risk_taker, risk_word, game_show, investment_allocation, 
                     market_follow, new_investment, buy_things, finance_reading,
                     previous_investments, investment_goal):
    """Save user profile information to the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        if not cursor.fetchone():
            conn.close()
            return False, "User does not exist"
        
        # Create profile data as JSON
        profile_data = {
            'risk_taker': risk_taker,
            'risk_word': risk_word,
            'game_show': game_show,
            'investment_allocation': investment_allocation,
            'market_follow': market_follow,
            'new_investment': new_investment,
            'buy_things': buy_things,
            'finance_reading': finance_reading,
            'previous_investments': previous_investments,
            'investment_goal': investment_goal
        }
        
        # Convert to JSON string
        profile_json = json.dumps(profile_data)
        
        # Check if profile already exists for this user
        cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
        existing_profile = cursor.fetchone()
        
        if existing_profile:
            # Update existing profile
            cursor.execute(
                'UPDATE user_profiles SET profile_data = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?',
                (profile_json, user_id)
            )
        else:
            # Insert new profile
            cursor.execute(
                'INSERT INTO user_profiles (user_id, profile_data) VALUES (?, ?)',
                (user_id, profile_json)
            )
        
        conn.commit()
        conn.close()
        return True, "Profile saved successfully"
    except Exception as e:
        return False, f"Profile save error: {str(e)}"

def get_user_profile(user_id):
    """Retrieve user profile information from the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
        profile = cursor.fetchone()
        
        if not profile:
            conn.close()
            return False, "Profile not found"
        
        profile_data = json.loads(profile['profile_data'])
        conn.close()
        return True, profile_data
    except Exception as e:
        return False, f"Error retrieving profile: {str(e)}"

# Initialize the database when this module is imported
initialize_db() 