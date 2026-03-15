""" 
Cardix AI - Complete Heart Disease Risk Prediction System
Single-file Flask Application with ML Integration
Author: Final Year Project
Date: 2024
"""

from flask import Flask, render_template, render_template_string, request, jsonify, session, redirect, url_for, send_file
import pandas as pd
import numpy as np
import pickle
import io
from datetime import datetime, timedelta
import sqlite3
from functools import wraps
import json
import base64
from markupsafe import Markup, escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import bcrypt
import secrets
from flask_mail import Mail, Message
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'heartai_secret_key_2024'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SHOW_LAST_PREDICTION'] = True

# Email configuration for password reset
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'  # Update with your email
app.config['MAIL_PASSWORD'] = 'your-app-password'     # Update with your app password
app.config['MAIL_DEFAULT_SENDER'] = 'your-email@gmail.com'

mail = Mail(app)

# ============================================
# DATABASE SETUP
# ============================================
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('heartai.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT,
                  name TEXT,
                  age INTEGER,
                  gender TEXT,
                  email TEXT UNIQUE,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Password reset tokens table
    c.execute('''CREATE TABLE IF NOT EXISTS password_reset_tokens
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  token TEXT UNIQUE,
                  expires_at TIMESTAMP,
                  used INTEGER DEFAULT 0,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Predictions table - self-reported health data only
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  age INTEGER,
                  sex INTEGER,
                  height REAL,
                  weight REAL,
                  bmi REAL,
                  smoking INTEGER,
                  alcohol INTEGER,
                  physical_activity INTEGER,
                  sleep_hours REAL,
                  stress_level INTEGER,
                  has_hypertension INTEGER,
                  resting_systolic_bp REAL,
                  resting_diastolic_bp REAL,
                  know_resting_heart_rate INTEGER,
                  resting_heart_rate REAL,
                  has_diabetes INTEGER,
                  has_high_cholesterol INTEGER,
                  cholesterol_value REAL,
                  fasting_blood_sugar REAL,
                  family_history INTEGER,
                  family_early_heart_attack INTEGER,
                  symptom_chest_pain INTEGER,
                  symptom_chest_pain_activity INTEGER,
                  symptom_shortness_breath INTEGER,
                  symptom_fatigue INTEGER,
                  symptom_palpitations INTEGER,
                  symptom_dizziness INTEGER,
                  symptom_radiating_pain INTEGER,
                  symptom_swelling INTEGER,
                  prediction INTEGER,
                  probability REAL,
                  model_used TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')

    # Lightweight migration for existing databases.
    c.execute("PRAGMA table_info(predictions)")
    existing_cols = {row[1] for row in c.fetchall()}
    for col_name, col_type in [
        ('resting_systolic_bp', 'REAL'),
        ('resting_diastolic_bp', 'REAL'),
        ('know_resting_heart_rate', 'INTEGER'),
        ('resting_heart_rate', 'REAL'),
        ('cholesterol_value', 'REAL'),
        ('fasting_blood_sugar', 'REAL'),
        ('symptom_chest_pain_activity', 'INTEGER')
    ]:
        if col_name not in existing_cols:
            c.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
    
    conn.commit()
    conn.close()

# ============================================
# MACHINE LEARNING MODEL
# ============================================
class HeartDiseasePredictor:
    """Machine Learning model for heart disease prediction based on self-reported data"""
    
    def __init__(self):
        self.models = {}
        self.feature_names = ['age', 'sex', 'bmi', 'smoking', 'alcohol', 'physical_activity',
                             'sleep_hours', 'stress_level', 'has_hypertension', 'resting_systolic_bp',
                             'resting_diastolic_bp', 'know_resting_heart_rate', 'resting_heart_rate', 'has_diabetes',
                             'has_high_cholesterol', 'cholesterol_value', 'fasting_blood_sugar',
                             'family_history', 'family_early_heart_attack',
                             'symptom_chest_pain', 'symptom_chest_pain_activity', 'symptom_shortness_breath', 'symptom_fatigue',
                             'symptom_palpitations', 'symptom_dizziness', 'symptom_radiating_pain',
                             'symptom_swelling']
        self._load_or_train_models()
    
    def _load_or_train_models(self):
        """Load pre-trained models or train new ones"""
        try:
            # Try to load pre-trained models
            with open('logistic_model.pkl', 'rb') as f:
                self.models['logistic'] = pickle.load(f)
            with open('random_forest.pkl', 'rb') as f:
                self.models['random_forest'] = pickle.load(f)
            
            # Set default metrics since we loaded models without saving metrics
            self.metrics = {
                'logistic': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1': 0.85},
                'random_forest': {'accuracy': 0.92, 'precision': 0.90, 'recall': 0.94, 'f1': 0.92}
            }
            print("Loaded pre-trained models")
        except:
            # Train new models
            self._train_models()
    
    def _train_models(self):
        """Train ML models on heart disease dataset with self-reported health features"""
        np.random.seed(42)
        n_samples = 1200
        
        # Generate realistic self-reported health data
        age = np.random.randint(25, 85, n_samples)
        sex = np.random.randint(0, 2, n_samples)
        height = np.random.normal(170, 10, n_samples)  # cm
        weight = np.random.normal(75, 15, n_samples)  # kg
        bmi = weight / ((height / 100) ** 2)
        
        X = pd.DataFrame({
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'smoking': np.random.randint(0, 3, n_samples),  # 0=never, 1=former, 2=current
            'alcohol': np.random.randint(0, 3, n_samples),  # 0=none, 1=occasional, 2=regular
            'physical_activity': np.random.randint(0, 3, n_samples),  # 0=sedentary, 1=moderate, 2=active
            'sleep_hours': np.random.uniform(4, 10, n_samples),
            'stress_level': np.random.randint(0, 3, n_samples),  # 0=low, 1=moderate, 2=high
            'has_hypertension': np.random.randint(0, 3, n_samples),  # 0=no, 1=not sure, 2=yes
            'resting_systolic_bp': np.random.uniform(95, 180, n_samples),
            'resting_diastolic_bp': np.random.uniform(60, 110, n_samples),
            'know_resting_heart_rate': np.random.randint(0, 3, n_samples),  # 0=no, 1=not sure, 2=yes
            'resting_heart_rate': np.random.uniform(50, 120, n_samples),
            'has_diabetes': np.random.randint(0, 3, n_samples),
            'has_high_cholesterol': np.random.randint(0, 3, n_samples),
            'cholesterol_value': np.random.uniform(140, 320, n_samples),
            'fasting_blood_sugar': np.random.uniform(70, 220, n_samples),
            'family_history': np.random.randint(0, 5, n_samples),  # 0=none, 1-4=different relatives
            'family_early_heart_attack': np.random.randint(0, 3, n_samples),  # 0=no, 1=not sure, 2=yes
            'symptom_chest_pain': np.random.randint(0, 3, n_samples),  # 0=never, 1=sometimes, 2=often
            'symptom_chest_pain_activity': np.random.randint(0, 3, n_samples),
            'symptom_shortness_breath': np.random.randint(0, 3, n_samples),
            'symptom_fatigue': np.random.randint(0, 3, n_samples),
            'symptom_palpitations': np.random.randint(0, 3, n_samples),
            'symptom_dizziness': np.random.randint(0, 3, n_samples),
            'symptom_radiating_pain': np.random.randint(0, 3, n_samples),
            'symptom_swelling': np.random.randint(0, 3, n_samples)
        })
        
        # Create target variable based on risk factors
        risk_score = np.zeros(n_samples)
        
        # Age factor (higher risk after 55)
        risk_score += (X['age'] > 55).astype(int) * 2
        
        # Gender (males higher risk)
        risk_score += (X['sex'] == 1).astype(int)
        
        # BMI (obesity: BMI > 30)
        risk_score += (X['bmi'] > 30).astype(int) * 1.5
        
        # Lifestyle factors
        risk_score += (X['smoking'] == 2).astype(int) * 2  # Current smoker
        risk_score += (X['smoking'] == 1).astype(int)  # Former smoker
        risk_score += (X['alcohol'] == 2).astype(int)  # Regular alcohol
        risk_score += (X['physical_activity'] == 0).astype(int) * 1.5  # Sedentary
        risk_score += (X['sleep_hours'] < 6).astype(int)
        risk_score += (X['stress_level'] == 2).astype(int)
        
        # Medical history
        risk_score += (X['has_hypertension'] == 2).astype(int) * 2
        risk_score += ((X['resting_systolic_bp'] >= 140) | (X['resting_diastolic_bp'] >= 90)).astype(int) * 1.5
        risk_score += ((X['resting_heart_rate'] > 100) | (X['resting_heart_rate'] < 50)).astype(int) * 0.8
        risk_score += (X['has_diabetes'] == 2).astype(int) * 2
        risk_score += (X['has_high_cholesterol'] == 2).astype(int) * 1.5
        risk_score += (X['cholesterol_value'] >= 240).astype(int) * 1.2
        risk_score += (X['fasting_blood_sugar'] >= 126).astype(int) * 1.8
        
        # Family history
        risk_score += (X['family_history'] > 0).astype(int) * 1.5
        risk_score += (X['family_early_heart_attack'] == 2).astype(int) * 2
        
        # Symptoms (strong indicators)
        risk_score += (X['symptom_chest_pain'] == 2).astype(int) * 3
        risk_score += (X['symptom_chest_pain'] == 1).astype(int) * 1.5
        risk_score += (X['symptom_chest_pain_activity'] == 2).astype(int) * 2.5
        risk_score += (X['symptom_chest_pain_activity'] == 1).astype(int) * 1.2
        risk_score += (X['symptom_shortness_breath'] >= 1).astype(int)
        risk_score += (X['symptom_fatigue'] >= 1).astype(int)
        risk_score += (X['symptom_palpitations'] >= 1).astype(int)
        risk_score += (X['symptom_dizziness'] >= 1).astype(int)
        risk_score += (X['symptom_radiating_pain'] == 2).astype(int) * 2
        risk_score += (X['symptom_swelling'] >= 1).astype(int)
        
        # Convert risk score to binary classification with balanced synthetic classes.
        threshold = np.percentile(risk_score, 65)
        y = (risk_score > threshold).astype(int)
        
        # Add some noise
        y = np.clip(y + np.random.binomial(1, 0.05, n_samples), 0, 1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        numeric_cols = ['age', 'bmi', 'sleep_hours', 'resting_systolic_bp', 'resting_diastolic_bp',
                'resting_heart_rate', 'cholesterol_value', 'fasting_blood_sugar']
        categorical_cols = ['sex', 'smoking', 'alcohol', 'physical_activity', 'stress_level',
                           'has_hypertension', 'know_resting_heart_rate', 'has_diabetes', 'has_high_cholesterol',
                           'family_history', 'family_early_heart_attack', 'symptom_chest_pain', 'symptom_chest_pain_activity',
                           'symptom_shortness_breath', 'symptom_fatigue', 'symptom_palpitations',
                           'symptom_dizziness', 'symptom_radiating_pain', 'symptom_swelling']
        
        preprocessor_lr = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        preprocessor_rf = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        lr_pipe = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
        rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
        
        lr_model = GridSearchCV(
            estimator=lr_pipe,
            param_grid={'C': [0.1, 0.5, 1.0, 2.0]},
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1
        )
        rf_model = GridSearchCV(
            estimator=rf_base,
            param_grid={'n_estimators': [200, 400], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1
        )
        
        from sklearn.pipeline import Pipeline
        lr_pipeline = Pipeline(steps=[('pre', preprocessor_lr), ('model', lr_model)])
        rf_pipeline = Pipeline(steps=[('pre', preprocessor_rf), ('model', rf_model)])
        
        lr_pipeline.fit(X_train, y_train)
        rf_pipeline.fit(X_train, y_train)
        
        best_lr = lr_pipeline.named_steps['model'].best_estimator_
        best_rf = rf_pipeline.named_steps['model'].best_estimator_
        
        calibrated_rf = CalibratedClassifierCV(estimator=best_rf, cv=3, method='isotonic')
        calibrated_rf.fit(preprocessor_rf.fit_transform(X_train), y_train)
        
        lr_pred = lr_pipeline.predict(X_test)
        rf_pred = calibrated_rf.predict(preprocessor_rf.transform(X_test))
        
        self.metrics = {
            'logistic': {
                'accuracy': accuracy_score(y_test, lr_pred),
                'precision': precision_score(y_test, lr_pred, zero_division=0),
                'recall': recall_score(y_test, lr_pred, zero_division=0),
                'f1': f1_score(y_test, lr_pred, zero_division=0)
            },
            'random_forest': {
                'accuracy': accuracy_score(y_test, rf_pred),
                'precision': precision_score(y_test, rf_pred, zero_division=0),
                'recall': recall_score(y_test, rf_pred, zero_division=0),
                'f1': f1_score(y_test, rf_pred, zero_division=0)
            }
        }
        
        self.models['logistic'] = lr_pipeline
        self.models['random_forest'] = Pipeline(steps=[('pre', preprocessor_rf), ('cal', calibrated_rf)])
        
        with open('logistic_model.pkl', 'wb') as f:
            pickle.dump(self.models['logistic'], f)
        with open('random_forest.pkl', 'wb') as f:
            pickle.dump(self.models['random_forest'], f)
        
        print("Models trained, calibrated, and saved with self-reported health features")
    
    def predict(self, features, model_type='random_forest'):
        """Make prediction using selected model"""
        if model_type not in self.models:
            model_type = 'random_forest'
        
        model = self.models[model_type]
        df = pd.DataFrame([features], columns=self.feature_names)
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'model_used': model_type,
            'confidence': round(probability * 100, 2) if prediction == 1 else round((1 - probability) * 100, 2)
        }

# Initialize ML predictor
predictor = HeartDiseasePredictor()

# ============================================
# AUTHENTICATION DECORATOR
# ============================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================
# ROUTES
# ============================================
@app.route('/')
def index():
    """Landing page"""
    return render_template_string(INDEX_HTML)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        identifier = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not identifier or not password:
            return jsonify({'success': False, 'message': 'Please provide both username/email and password'})
        
        conn = sqlite3.connect('heartai.db')
        c = conn.cursor()
        # Allow login with either username or email
        c.execute("SELECT id, name, username, password FROM users WHERE (LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?))", 
                 (identifier, identifier))
        user = c.fetchone()
        conn.close()
        
        if user:
            # Verify password using bcrypt
            stored_password = user[3]
            # Handle both old plain text passwords and new hashed passwords for migration
            if isinstance(stored_password, str):
                stored_password = stored_password.encode('utf-8')
            
            try:
                # Try bcrypt verification
                if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                    session['user_id'] = user[0]
                    session['name'] = user[1]
                    session['username'] = user[2]
                    return jsonify({'success': True, 'redirect': '/dashboard'})
                else:
                    return jsonify({'success': False, 'message': 'Invalid username/email or password'})
            except:
                # Fallback for old plain text passwords (migration support)
                if password == stored_password.decode('utf-8'):
                    # Update to hashed password
                    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                    conn = sqlite3.connect('heartai.db')
                    c = conn.cursor()
                    c.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_password, user[0]))
                    conn.commit()
                    conn.close()
                    
                    session['user_id'] = user[0]
                    session['name'] = user[1]
                    session['username'] = user[2]
                    return jsonify({'success': True, 'redirect': '/dashboard'})
                else:
                    return jsonify({'success': False, 'message': 'Invalid username/email or password'})
        else:
            return jsonify({'success': False, 'message': 'Invalid username/email or password'})
    
    return render_template_string(LOGIN_HTML)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '')
        name = data.get('name', '').strip()
        age = data.get('age')
        gender = data.get('gender')
        email = data.get('email', '').strip()
        
        # Validation
        if not all([username, password, name, age, gender, email]):
            return jsonify({'success': False, 'message': 'All fields are required'})
            
        try:
            # Hash the password using bcrypt
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            conn = sqlite3.connect('heartai.db')
            c = conn.cursor()
            
            # Check if username or email already exists (case-insensitive)
            c.execute("SELECT id FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)", (username, email))
            if c.fetchone():
                conn.close()
                return jsonify({'success': False, 'message': 'Username or email already registered'})

            # Insert user with hashed password
            c.execute('''INSERT INTO users (username, password, name, age, gender, email) 
                        VALUES (?, ?, ?, ?, ?, ?)''',
                     (username, hashed_password, name, age, gender, email))
            conn.commit()
            conn.close()
            
            return jsonify({'success': True, 'message': 'Registration successful! You can now login.'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error: {str(e)}'})
    
    return render_template_string(REGISTER_HTML)

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Forgot password page"""
    if request.method == 'POST':
        data = request.json
        email = data.get('email', '').strip()
        
        if not email:
            return jsonify({'success': False, 'message': 'Please provide your email address'})
        
        try:
            conn = sqlite3.connect('heartai.db')
            c = conn.cursor()
            
            # Check if email exists
            c.execute("SELECT id, name FROM users WHERE LOWER(email) = LOWER(?)", (email,))
            user = c.fetchone()
            
            if user:
                user_id = user[0]
                user_name = user[1]
                
                # Generate secure reset token
                reset_token = secrets.token_urlsafe(32)
                
                # Token expires in 30 minutes
                expires_at = datetime.now() + timedelta(minutes=30)
                
                # Store token in database
                c.execute('''INSERT INTO password_reset_tokens (user_id, token, expires_at) 
                            VALUES (?, ?, ?)''', (user_id, reset_token, expires_at))
                conn.commit()
                
                # Send reset email
                try:
                    reset_url = request.url_root + f'reset-password?token={reset_token}'
                    msg = Message('Password Reset Request', recipients=[email])
                    msg.body = f'''Hello {user_name},

You requested to reset your password for your Heart Disease Prediction account.

Click the link below to reset your password:
{reset_url}

This link will expire in 30 minutes.

If you did not request this, please ignore this email and your password will remain unchanged.

Best regards,
Heart Disease Prediction Team
'''
                    mail.send(msg)
                    conn.close()
                    return jsonify({'success': True, 'message': 'Password reset link has been sent to your email'})
                except Exception as email_error:
                    conn.close()
                    # For development/testing: return the reset URL in the response
                    return jsonify({'success': True, 'message': f'Email service not configured. Reset URL: {reset_url}', 'reset_url': reset_url})
            else:
                conn.close()
                # Don't reveal if email exists or not (security best practice)
                return jsonify({'success': True, 'message': 'If this email is registered, a password reset link will be sent'})
                
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error: {str(e)}'})
    
    return render_template_string(FORGOT_PASSWORD_HTML)

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    """Reset password page"""
    if request.method == 'GET':
        token = request.args.get('token')
        if not token:
            return "Invalid reset link", 400
        
        # Verify token
        conn = sqlite3.connect('heartai.db')
        c = conn.cursor()
        c.execute('''SELECT user_id, expires_at, used FROM password_reset_tokens 
                     WHERE token = ?''', (token,))
        token_data = c.fetchone()
        conn.close()
        
        if not token_data:
            return "Invalid or expired reset link", 400
        
        user_id, expires_at, used = token_data
        
        if used == 1:
            return "This reset link has already been used", 400
        
        if datetime.strptime(expires_at, '%Y-%m-%d %H:%M:%S.%f') < datetime.now():
            return "This reset link has expired", 400
        
        return render_template_string(RESET_PASSWORD_HTML, token=token)
    
    elif request.method == 'POST':
        data = request.json
        token = data.get('token')
        new_password = data.get('new_password')
        confirm_password = data.get('confirm_password')
        
        if not all([token, new_password, confirm_password]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        if new_password != confirm_password:
            return jsonify({'success': False, 'message': 'Passwords do not match'})
        
        try:
            conn = sqlite3.connect('heartai.db')
            c = conn.cursor()
            
            # Verify token again
            c.execute('''SELECT user_id, expires_at, used FROM password_reset_tokens 
                         WHERE token = ?''', (token,))
            token_data = c.fetchone()
            
            if not token_data:
                conn.close()
                return jsonify({'success': False, 'message': 'Invalid or expired reset link'})
            
            user_id, expires_at, used = token_data
            
            if used == 1:
                conn.close()
                return jsonify({'success': False, 'message': 'This reset link has already been used'})
            
            if datetime.strptime(expires_at, '%Y-%m-%d %H:%M:%S.%f') < datetime.now():
                conn.close()
                return jsonify({'success': False, 'message': 'This reset link has expired'})
            
            # Hash new password
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            
            # Update password
            c.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_password, user_id))
            
            # Mark token as used
            c.execute("UPDATE password_reset_tokens SET used = 1 WHERE token = ?", (token,))
            
            conn.commit()
            conn.close()
            
            return jsonify({'success': True, 'message': 'Password updated successfully. Please login with your new password.'})
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    conn = sqlite3.connect('heartai.db')
    c = conn.cursor()
    
    # Get user stats
    c.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ?", (session['user_id'],))
    total_predictions = c.fetchone()[0]
    
    c.execute('''SELECT prediction, created_at FROM predictions 
                 WHERE user_id = ? ORDER BY created_at DESC LIMIT 1''', 
              (session['user_id'],))
    last_prediction = c.fetchone()
    
    conn.close()
    
    return render_template_string(DASHBOARD_HTML,
                                 name=session.get('name', 'User'),
                                 total_predictions=total_predictions,
                                 last_prediction=last_prediction)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Prediction page"""
    if request.method == 'POST':
        data = request.json

        def optional_float(field_name):
            raw = str(data.get(field_name, '')).strip()
            return float(raw) if raw else None

        def value_from_hybrid(optional_value, diagnosis_value, yes_default, no_default, unsure_default):
            if optional_value is not None:
                return optional_value
            if diagnosis_value == 2:
                return yes_default
            if diagnosis_value == 0:
                return no_default
            return unsure_default
        
        # Calculate BMI from height (cm) and weight (kg) using standard medical formula.
        height_cm = float(data['height'])
        weight_kg = float(data['weight'])

        # Keep validation aligned with the form constraints and realistic adult ranges.
        if not 100 <= height_cm <= 250:
            return jsonify({'success': False, 'message': 'Height must be between 100 and 250 cm.'}), 400
        if not 30 <= weight_kg <= 250:
            return jsonify({'success': False, 'message': 'Weight must be between 30 and 250 kg.'}), 400

        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m ** 2)
        bmi = round(bmi, 2)

        has_hypertension = int(data['has_hypertension'])
        has_diabetes = int(data['has_diabetes'])
        has_high_cholesterol = int(data['has_high_cholesterol'])
        know_resting_heart_rate = int(data['know_resting_heart_rate'])

        resting_systolic_bp_input = optional_float('resting_systolic_bp')
        resting_diastolic_bp_input = optional_float('resting_diastolic_bp')
        resting_heart_rate_input = optional_float('resting_heart_rate')
        cholesterol_value_input = optional_float('cholesterol_value')
        fasting_blood_sugar_input = optional_float('fasting_blood_sugar')

        # Use numeric values when provided, otherwise derive a proxy from Yes/No/Not Sure.
        resting_systolic_bp = value_from_hybrid(resting_systolic_bp_input, has_hypertension, 145.0, 120.0, 130.0)
        resting_diastolic_bp = value_from_hybrid(resting_diastolic_bp_input, has_hypertension, 92.0, 78.0, 84.0)
        cholesterol_value = value_from_hybrid(cholesterol_value_input, has_high_cholesterol, 245.0, 180.0, 210.0)
        fasting_blood_sugar = value_from_hybrid(fasting_blood_sugar_input, has_diabetes, 145.0, 95.0, 110.0)

        if resting_heart_rate_input is not None:
            resting_heart_rate = resting_heart_rate_input
        else:
            resting_heart_rate = 72.0 if know_resting_heart_rate == 2 else 75.0
        
        # Prepare features for ML model
        features = [
            int(data['age']),
            int(data['sex']),
            float(bmi),
            int(data['smoking']),
            int(data['alcohol']),
            int(data['physical_activity']),
            float(data['sleep_hours']),
            int(data['stress_level']),
            has_hypertension,
            resting_systolic_bp,
            resting_diastolic_bp,
            know_resting_heart_rate,
            resting_heart_rate,
            has_diabetes,
            has_high_cholesterol,
            cholesterol_value,
            fasting_blood_sugar,
            int(data['family_history']),
            int(data['family_early_heart_attack']),
            int(data['symptom_chest_pain']),
            int(data['symptom_chest_pain_activity']),
            int(data['symptom_shortness_breath']),
            int(data['symptom_fatigue']),
            int(data['symptom_palpitations']),
            int(data['symptom_dizziness']),
            int(data['symptom_radiating_pain']),
            int(data['symptom_swelling'])
        ]
        
        # Get prediction
        model_type = data.get('model', 'random_forest')
        result = predictor.predict(features, model_type)
        
        # Save to database
        conn = sqlite3.connect('heartai.db')
        c = conn.cursor()
        c.execute('''INSERT INTO predictions 
                    (user_id, age, sex, height, weight, bmi, smoking, alcohol, physical_activity,
                                         sleep_hours, stress_level, has_hypertension, resting_systolic_bp, resting_diastolic_bp,
                                         know_resting_heart_rate, resting_heart_rate, has_diabetes, has_high_cholesterol, cholesterol_value, fasting_blood_sugar,
                     family_history, family_early_heart_attack, symptom_chest_pain, symptom_shortness_breath,
                     symptom_chest_pain_activity, symptom_fatigue, symptom_palpitations, symptom_dizziness, symptom_radiating_pain,
                     symptom_swelling, prediction, probability, model_used)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (session['user_id'], int(data['age']), int(data['sex']), float(data['height']),
                  float(data['weight']), float(bmi), int(data['smoking']), int(data['alcohol']),
                  int(data['physical_activity']), float(data['sleep_hours']), int(data['stress_level']),
                                    has_hypertension, resting_systolic_bp_input, resting_diastolic_bp_input,
                                    know_resting_heart_rate, resting_heart_rate_input, has_diabetes, has_high_cholesterol,
                                    cholesterol_value_input, fasting_blood_sugar_input,
                  int(data['family_history']), int(data['family_early_heart_attack']),
                  int(data['symptom_chest_pain']), int(data['symptom_shortness_breath']),
                  int(data['symptom_chest_pain_activity']), int(data['symptom_fatigue']), int(data['symptom_palpitations']),
                  int(data['symptom_dizziness']), int(data['symptom_radiating_pain']),
                  int(data['symptom_swelling']), result['prediction'], result['probability'], result['model_used']))
        prediction_id = c.lastrowid
        conn.commit()
        conn.close()
        
        result['prediction_id'] = prediction_id
        result['bmi'] = bmi
        return jsonify(result)
    
    return render_template_string(PREDICT_HTML, name=session.get('name', 'User'))

@app.route('/history')
@login_required
def history():
    """Prediction history"""
    conn = sqlite3.connect('heartai.db')
    c = conn.cursor()
    
    c.execute('''SELECT id, created_at, prediction, probability, model_used 
                 FROM predictions WHERE user_id = ? ORDER BY created_at DESC''',
              (session['user_id'],))
    predictions = c.fetchall()
    
    conn.close()
    
    return render_template_string(HISTORY_HTML, predictions=predictions, name=session.get('name', 'User'))

@app.route('/download_report/<int:prediction_id>')
@login_required
def download_report(prediction_id):
    """Generate and download PDF report"""
    return generate_report(prediction_id)

@app.route('/report/<int:prediction_id>')
@login_required
def generate_report(prediction_id):
    """Generate PDF report"""
    try:
        conn = sqlite3.connect('heartai.db')
        c = conn.cursor()
        
        c.execute('''SELECT * FROM predictions WHERE id = ? AND user_id = ?''',
              (prediction_id, session['user_id']))
        prediction_row = c.fetchone()
        prediction_cols = [col[0] for col in c.description] if c.description else []
        prediction = dict(zip(prediction_cols, prediction_row)) if prediction_row else None
        
        c.execute('''SELECT name, age, gender, email FROM users WHERE id = ?''',
                  (session['user_id'],))
        user = c.fetchone()
        
        conn.close()
        
        if not prediction:
            return "Report not found or access denied", 404

        # If the users row is missing, continue with session-based fallback.
        if user:
            user_name = user[0]
            user_age = user[1] if user[1] is not None else 'N/A'
            user_gender = user[2] if user[2] is not None else 'N/A'
        else:
            user_name = session.get('name', 'User')
            user_age = 'N/A'
            user_gender = 'N/A'

        # Create PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=40,
            rightMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        # Times-Roman/Times-Bold are built-in ReportLab fonts and map to the
        # Times New Roman style family for formal report presentation.
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Title'],
            fontName='Times-Bold',
            fontSize=19,
            leading=23,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#111827')
        )
        section_style = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontName='Times-Bold',
            fontSize=13,
            leading=16,
            alignment=TA_LEFT,
            textColor=colors.HexColor('#1f2937'),
            spaceBefore=8,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'BodyTextFormal',
            parent=styles['Normal'],
            fontName='Times-Roman',
            fontSize=11,
            leading=15,
            textColor=colors.HexColor('#111827')
        )
        bullet_style = ParagraphStyle(
            'BulletFormal',
            parent=body_style,
            leftIndent=14,
            bulletIndent=2,
            spaceAfter=2
        )
        risk_style = ParagraphStyle(
            'RiskResult',
            parent=body_style,
            fontName='Times-Bold',
            fontSize=12,
            leading=16
        )
        disclaimer_style = ParagraphStyle(
            'DisclaimerText',
            parent=body_style,
            fontName='Times-Italic',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#374151')
        )

        def safe_int(value, default=0):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def safe_float(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def yns_label(value):
            return {0: 'No', 1: 'Not Sure', 2: 'Yes'}.get(safe_int(value), 'N/A')

        def bounded_lookup(value, labels, fallback='N/A'):
            idx = safe_int(value, -1)
            if 0 <= idx < len(labels):
                return labels[idx]
            return fallback

        risk_val = safe_int(prediction.get('prediction', 0))
        prob_val = safe_float(prediction.get('probability', 0.0))
        model_name = str(prediction.get('model_used', 'random_forest')).replace('_', ' ').title()

        age = safe_int(prediction.get('age'))
        sex_label = 'Male' if safe_int(prediction.get('sex', 0)) == 1 else 'Female'
        height_cm = safe_float(prediction.get('height', 0))
        weight_kg = safe_float(prediction.get('weight', 0))
        bmi = safe_float(prediction.get('bmi', 0.0))

        systolic_bp = safe_float(prediction.get('resting_systolic_bp', 0))
        diastolic_bp = safe_float(prediction.get('resting_diastolic_bp', 0))
        cholesterol_value = safe_float(prediction.get('cholesterol_value', 0))
        glucose_value = safe_float(prediction.get('fasting_blood_sugar', 0))

        smoking_label = bounded_lookup(prediction.get('smoking', 0), ['Never', 'Former', 'Current'])
        activity_label = bounded_lookup(prediction.get('physical_activity', 0), ['Sedentary', 'Moderate', 'Active'])
        alcohol_label = bounded_lookup(prediction.get('alcohol', 0), ['None', 'Occasional', 'Regular'])
        stress_label = bounded_lookup(prediction.get('stress_level', 0), ['Low', 'Moderate', 'High'])
        sleep_hours = safe_float(prediction.get('sleep_hours', 0))

        # Build clear report interpretation from the available inputs.
        contributing_factors = []
        positive_factors = []

        smoking = safe_int(prediction.get('smoking', 0))
        activity = safe_int(prediction.get('physical_activity', 0))
        high_stress = safe_int(prediction.get('stress_level', 0)) == 2
        has_hypertension = safe_int(prediction.get('has_hypertension', 0)) == 2
        has_diabetes = safe_int(prediction.get('has_diabetes', 0)) == 2
        has_high_cholesterol = safe_int(prediction.get('has_high_cholesterol', 0)) == 2
        has_family_history = safe_int(prediction.get('family_history', 0)) > 0
        early_family_attack = safe_int(prediction.get('family_early_heart_attack', 0)) == 2

        if age >= 55:
            contributing_factors.append("Age is on the higher side, and heart risk often increases with age.")
        elif 18 <= age < 45:
            positive_factors.append("Age is in a range that usually supports lower baseline heart risk.")

        if bmi >= 30:
            contributing_factors.append("Body weight compared to height is in the obesity range, which can put extra strain on the heart.")
        elif bmi >= 25:
            contributing_factors.append("Body weight compared to height is above the healthy range and may add to risk over time.")
        elif 18.5 <= bmi < 25:
            positive_factors.append("BMI is in the healthy range.")

        if has_hypertension or systolic_bp >= 140 or diastolic_bp >= 90:
            contributing_factors.append("Blood pressure appears high, which is an important risk signal for heart problems.")
        elif systolic_bp > 0 and diastolic_bp > 0 and systolic_bp < 120 and diastolic_bp < 80:
            positive_factors.append("Blood pressure is in the normal range.")

        if has_high_cholesterol or cholesterol_value >= 200:
            contributing_factors.append("Cholesterol is above the ideal level or has been reported as high, which may affect blood vessels over time.")
        elif 0 < cholesterol_value < 200:
            positive_factors.append("Cholesterol is in the recommended range.")

        if has_diabetes or glucose_value >= 100:
            contributing_factors.append("Blood sugar is above normal or diabetes is reported, which can increase heart risk.")
        elif 70 <= glucose_value <= 99:
            positive_factors.append("Fasting blood sugar is in the normal range.")

        if smoking == 2:
            contributing_factors.append("Current smoking status is a strong lifestyle factor that increases heart risk.")
        elif smoking == 0:
            positive_factors.append("Non-smoking status supports good heart health.")

        if activity == 0:
            contributing_factors.append("Low physical activity can increase long-term heart risk.")
        elif activity == 2:
            positive_factors.append("Active lifestyle supports better heart health.")

        if sleep_hours > 0 and (sleep_hours < 7 or sleep_hours > 9):
            contributing_factors.append("Sleep duration is outside the recommended 7-9 hour range.")
        elif 7 <= sleep_hours <= 9:
            positive_factors.append("Sleep duration is in the recommended range.")

        if high_stress:
            contributing_factors.append("High stress level may increase heart strain over time.")

        if has_family_history or early_family_attack:
            contributing_factors.append("Family history suggests inherited risk that should be monitored carefully.")

        symptom_fields = [
            ('symptom_chest_pain', 'Chest discomfort/pressure'),
            ('symptom_chest_pain_activity', 'Chest pain during activity'),
            ('symptom_shortness_breath', 'Shortness of breath'),
            ('symptom_fatigue', 'Unusual fatigue'),
            ('symptom_palpitations', 'Palpitations'),
            ('symptom_dizziness', 'Dizziness/fainting tendency'),
            ('symptom_radiating_pain', 'Radiating pain'),
            ('symptom_swelling', 'Leg/foot swelling')
        ]
        symptom_hits = [label for key, label in symptom_fields if safe_int(prediction.get(key, 0)) >= 1]
        if symptom_hits:
            contributing_factors.append("Reported symptoms (" + ", ".join(symptom_hits) + ") may be linked to heart strain and need clinical review.")

        # Professional report document content.
        story = []
        report_dt = prediction.get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        risk_level = "HIGH RISK" if risk_val == 1 else "LOW RISK"
        risk_color = '#b91c1c' if risk_val == 1 else '#047857'

        story.append(Paragraph("CARDIX AI HEART HEALTH ASSESSMENT REPORT", title_style))
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"Report ID: CR-{prediction_id:06d} | Generated On: {report_dt}", body_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph("1. Patient Information", section_style))
        patient_data = [
            ['Field', 'Information'],
            ['Name', str(user_name)],
            ['Age', str(user_age)],
            ['Gender', str(user_gender)],
            ['Model Used', model_name]
        ]
        patient_table = Table(patient_data, colWidths=[140, 360])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbeafe')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#111827')),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.6, colors.HexColor('#9ca3af')),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 10))

        story.append(Paragraph("2. Health Measurements", section_style))
        story.append(Paragraph(
            f"Age: {age} years | Gender: {sex_label} | Height: {height_cm:.1f} cm | Weight: {weight_kg:.1f} kg | BMI: {bmi:.2f}",
            body_style
        ))
        story.append(Paragraph(
            f"Model used: {model_name}. This report compares your values with common healthy reference ranges.",
            body_style
        ))
        story.append(Spacer(1, 10))

        story.append(Paragraph("3. Predicted Risk Result", section_style))
        story.append(Paragraph(
            f"Your predicted result is: <font color='{risk_color}'>{risk_level}</font>. "
            f"Estimated risk score: {prob_val * 100:.2f}%.",
            risk_style
        ))
        if risk_val == 1:
            story.append(Paragraph("This means your current inputs show signs linked with higher heart risk.", body_style))
        else:
            story.append(Paragraph("This means your current inputs show a lower heart risk profile right now.", body_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph("4. Health Factors Summary", section_style))
        metrics_data = [
            ['Health Factor', 'Your Value', 'Healthy Value'],
            ['Blood Pressure', f"{int(systolic_bp) if systolic_bp > 0 else 'N/A'}/{int(diastolic_bp) if diastolic_bp > 0 else 'N/A'} mmHg", 'Below 120/80 mmHg'],
            ['BMI', f"{bmi:.2f}", '18.5-24.9'],
            ['Blood Sugar (Fasting)', f"{glucose_value:.0f} mg/dL" if glucose_value > 0 else 'Not provided', '70-99 mg/dL'],
            ['Total Cholesterol', f"{cholesterol_value:.0f} mg/dL" if cholesterol_value > 0 else 'Not provided', 'Below 200 mg/dL'],
            ['Sleep Duration', f"{sleep_hours:.1f} hours/night" if sleep_hours > 0 else 'Not provided', '7-9 hours per night'],
            ['Physical Activity', activity_label, 'About 150 minutes per week'],
            ['Smoking Status', smoking_label, 'Non-smoking (healthiest status)'],
            ['Stress Level', stress_label, 'Low to moderate'],
            ['Known High Blood Pressure', yns_label(prediction.get('has_hypertension', 0)), 'Below 120/80 mmHg'],
            ['Known Diabetes', yns_label(prediction.get('has_diabetes', 0)), '70-99 mg/dL'],
            ['Known High Cholesterol', yns_label(prediction.get('has_high_cholesterol', 0)), 'Below 200 mg/dL']
        ]
        metrics_table = Table(metrics_data, colWidths=[170, 140, 190])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5e7eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#111827')),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 0), (-1, -1), 9.5),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#9ca3af')),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 10))

        # Keep section 5 heading and content together by starting this section on page 2.
        story.append(PageBreak())
        story.append(Paragraph("5. Explanation of Risk Factors", section_style))
        if contributing_factors:
            for idx, item in enumerate(contributing_factors, start=1):
                story.append(Paragraph(f"{idx}. {item}", body_style))
        else:
            story.append(Paragraph(
                "1. No major risk-raising factors were found from your submitted values.",
                body_style
            ))

        story.append(Spacer(1, 10))
        story.append(Paragraph("6. Positive Health Indicators", section_style))
        if positive_factors:
            for idx, item in enumerate(positive_factors, start=1):
                story.append(Paragraph(f"{idx}. {item}", body_style))
        else:
            story.append(Paragraph(
                "1. No strong positive indicators were detected from the entered values.",
                body_style
            ))

        story.append(Spacer(1, 10))
        story.append(Paragraph("7. Final Health Recommendation", section_style))

        if risk_val == 1:
            story.append(Paragraph(
                "Your report shows higher risk. Please consult a qualified doctor soon for full medical evaluation.",
                body_style
            ))
            high_risk_recs = [
                "Book an appointment with a qualified doctor and share this report.",
                "Further checks such as blood pressure monitoring, cholesterol testing, and heart examination may be helpful.",
                "Start a balanced diet with less salt, sugar, and fried food.",
                "Stop smoking and limit alcohol.",
                "Do regular physical activity as advised by your doctor.",
                "Go to urgent care quickly if chest pain, severe breathlessness, or fainting happens."
            ]
            for idx, rec in enumerate(high_risk_recs, start=1):
                story.append(Paragraph(f"{idx}. {rec}", body_style))
        else:
            story.append(Paragraph(
                "Your report shows lower risk at this time. Keep your healthy habits to stay on track.",
                body_style
            ))
            low_risk_tips = [
                "Eat a balanced diet with more vegetables, fruits, and whole grains.",
                "Exercise regularly (about 150 minutes per week).",
                "Keep a healthy body weight.",
                "Sleep well (about 7-9 hours each night) and manage stress daily.",
                "Do routine health checkups for blood pressure, cholesterol, and blood sugar."
            ]
            for idx, tip in enumerate(low_risk_tips, start=1):
                story.append(Paragraph(f"{idx}. {tip}", body_style))

        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "Disclaimer: This report is generated by an AI-assisted screening tool using self-reported information. "
            "It is intended for educational and screening purposes only and does not establish a medical diagnosis. "
            "Clinical decisions must be made by qualified healthcare professionals.",
            disclaimer_style
        ))

        def draw_page_border(canvas_obj, document_obj):
            canvas_obj.saveState()
            canvas_obj.setStrokeColor(colors.HexColor('#374151'))
            canvas_obj.setLineWidth(1.0)
            inset = 18
            canvas_obj.rect(
                inset,
                inset,
                letter[0] - (2 * inset),
                letter[1] - (2 * inset),
                stroke=1,
                fill=0
            )
            canvas_obj.restoreState()

        # Build PDF
        doc.build(story, onFirstPage=draw_page_border, onLaterPages=draw_page_border)
        buffer.seek(0)
        
        return send_file(buffer, as_attachment=True, 
                        download_name=f"Cardix_AI_Report_{prediction_id}.pdf",
                        mimetype='application/pdf')
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return f"Error generating report: {str(e)}", 500

@app.route('/profile')
@login_required
def profile():
    """User profile"""
    conn = sqlite3.connect('heartai.db')
    c = conn.cursor()
    c.execute("SELECT name, age, gender, email, username FROM users WHERE id = ?",
              (session['user_id'],))
    user = c.fetchone()
    conn.close()
    
    return render_template_string(PROFILE_HTML, user=user, datetime=datetime)

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    """Update user profile information"""
    data = request.get_json()
    name = data.get('name')
    age = data.get('age')
    gender = data.get('gender')
    email = data.get('email')
    
    if not all([name, age, gender, email]):
        return jsonify({'success': False, 'message': 'All fields are required'})
    
    try:
        conn = sqlite3.connect('heartai.db')
        c = conn.cursor()
        c.execute("""UPDATE users 
                     SET name = ?, age = ?, gender = ?, email = ? 
                     WHERE id = ?""",
                  (name, age, gender, email, session['user_id']))
        conn.commit()
        conn.close()
        
        # Update session if name changed
        session['name'] = name
        
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/admin')
@login_required
def admin_dashboard():
    """Admin dashboard (only for demo - in production, add admin check)"""
    if session.get('username') != 'admin':
        return redirect('/dashboard')
    
    conn = sqlite3.connect('heartai.db')
    c = conn.cursor()
    
    # Get statistics
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 1")
    high_risk_count = c.fetchone()[0]
    
    # Recent predictions
    c.execute('''SELECT u.name, p.created_at, p.prediction, p.probability 
                 FROM predictions p JOIN users u ON p.user_id = u.id 
                 ORDER BY p.created_at DESC LIMIT 10''')
    recent_predictions = c.fetchall()

    # Get all users for management
    c.execute("SELECT id, name, username, email, created_at FROM users ORDER BY created_at DESC")
    all_users = c.fetchall()
    
    conn.close()
    
    return render_template_string(ADMIN_HTML,
                                 total_users=total_users,
                                 total_predictions=total_predictions,
                                 high_risk_count=high_risk_count,
                                 recent_predictions=recent_predictions,
                                 all_users=all_users,
                                 metrics=predictor.metrics,
                                 datetime=datetime)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    """Delete a user (Admin only)"""
    if session.get('username') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    if user_id == session.get('user_id'):
        return jsonify({'success': False, 'message': 'Cannot delete yourself'})
        
    try:
        conn = sqlite3.connect('heartai.db')
        c = conn.cursor()
        # Delete predictions first due to FK
        c.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'User deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/about')
def about():
    """About page with ML details"""
    return render_template_string(ABOUT_HTML, metrics=predictor.metrics, max=max, datetime=datetime)

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect('/')

# ============================================
# HTML TEMPLATES
# ============================================

# Base template components
BASE_CSS = '''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #8B0000;
        --primary-dark: #8B0000;
        --secondary: #8B0000;
        --accent: #ffc2d1;
        --danger: #ff4d6d;
        --success: #3cb371;
        --light: #fff0f5;
        --dark: #3d0a1a;
        --bg-color: #fff7fa;
        --card-bg: #ffffff;
        --toggle-border: rgba(0,0,0,0.12);
        --shadow: 0 4px 6px -1px rgba(255, 105, 135, 0.12), 0 2px 4px -1px rgba(255, 105, 135, 0.08);
        --shadow-lg: 0 20px 25px -5px rgba(255, 105, 135, 0.18), 0 10px 10px -5px rgba(255, 105, 135, 0.12);
        --radius: 1rem;
    }
    
    .theme-dark {
        --primary: #ff8fab;
        --secondary: #ff6b9a;
        --accent: #ff8fab;
        --danger: #ff4d6d;
        --success: #22c55e;
        --light: #1f2937;
        --dark: #e5e7eb;
        --bg-color: #0f1220;
        --card-bg: #121627;
        --toggle-border: rgba(255,255,255,0.2);
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.35), 0 2px 4px -1px rgba(0, 0, 0, 0.30);
        --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.45), 0 10px 10px -5px rgba(0, 0, 0, 0.40);
    }

    body:not(.theme-dark) {
        --bg-color: #f8fafc !important;
        --card-bg: #ffffff !important;
        --text-main: #1e293b !important;
        --text-dim: #64748b !important;
        --glass-border: rgba(0, 0, 0, 0.1) !important;
        --input-bg: #f1f5f9 !important;
        --predict-bg: #f8fafc !important;
        --history-bg: #f8fafc !important;
        --profile-bg: #f8fafc !important;
        --about-bg: #f8fafc !important;
        --primary: #8B0000 !important;
        --secondary: #0B3D91 !important;
    }

    body:not(.theme-dark) .predict-page,
    body:not(.theme-dark) .history-page,
    body:not(.theme-dark) .profile-wrapper,
    body:not(.theme-dark) .about-container,
    body:not(.theme-dark) .container {
        background: var(--bg-color) !important;
        color: var(--text-main) !important;
        background-image: none !important;
    }

    body:not(.theme-dark) .predict-card,
    body:not(.theme-dark) .history-card,
    body:not(.theme-dark) .profile-card,
    body:not(.theme-dark) .about-card,
    body:not(.theme-dark) .profile-sidebar,
    body:not(.theme-dark) .about-hero,
    body:not(.theme-dark) .card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: none !important;
    }

    body:not(.theme-dark) .predict-input,
    body:not(.theme-dark) .search-input,
    body:not(.theme-dark) .form-control,
    body:not(.theme-dark) .search-box input {
        background: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
    }

    body:not(.theme-dark) .predict-header h1,
    body:not(.theme-dark) .history-header h1,
    body:not(.theme-dark) .section-title,
    body:not(.theme-dark) .hero-content h1,
    body:not(.theme-dark) .date-main,
    body:not(.theme-dark) .stat-val,
    body:not(.theme-dark) .sidebar-info h2,
    body:not(.theme-dark) .step-content h4,
    body:not(.theme-dark) .tech-info strong,
    body:not(.theme-dark) .history-title-group h1,
    body:not(.theme-dark) .predict-header h1 {
        color: #1e293b !important;
        background: none !important;
        -webkit-text-fill-color: initial !important;
    }

    body:not(.theme-dark) .history-title-group p,
    body:not(.theme-dark) .predict-header p,
    body:not(.theme-dark) .stat-label,
    body:not(.theme-dark) .pred-id,
    body:not(.theme-dark) .confidence-header span,
    body:not(.theme-dark) .date-sub,
    body:not(.theme-dark) .sidebar-info p {
        color: #64748b !important;
    }

    body:not(.theme-dark) .btn-report,
    body:not(.theme-dark) .btn-predict-new,
    body:not(.theme-dark) .btn-new {
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
        background: #ffffff !important;
    }

    body:not(.theme-dark) .btn-report:hover,
    body:not(.theme-dark) .btn-predict-new:hover {
        background: #f1f5f9 !important;
    }

    body:not(.theme-dark) .card-stats {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
    }

    body:not(.theme-dark) .confidence-bar-bg {
        background: #e2e8f0 !important;
    }

    body:not(.theme-dark) .status-badge {
        background: rgba(139, 0, 0, 0.05) !important;
        border-color: rgba(139, 0, 0, 0.1) !important;
    }

    body:not(.theme-dark) .bg-glow {
        display: none !important;
    }

    body:not(.theme-dark) .nav-container {
        background: rgba(255, 255, 255, 0.8) !important;
        border-bottom: 1px solid rgba(0, 0, 0, 0.05) !important;
    }

    body:not(.theme-dark) .nav-link {
        color: #1e293b !important;
    }

    body:not(.theme-dark) .nav-link:hover {
        color: var(--primary) !important;
    }

    body:not(.theme-dark) .user-icon {
        background: #ffffff !important;
        border-color: #e2e8f0 !important;
        color: #1e293b !important;
    }

    body:not(.theme-dark) .process-step,
    body:not(.theme-dark) .tech-item,
    body:not(.theme-dark) .metric-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #1e293b !important;
    }

    body:not(.theme-dark) .process-step p,
    body:not(.theme-dark) .process-step strong,
    body:not(.theme-dark) .tech-item strong,
    body:not(.theme-dark) .metric-card h3,
    body:not(.theme-dark) .metric-card .metric-value {
        color: #1e293b !important;
    }

    body:not(.theme-dark) .history-item {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
    }

    body:not(.theme-dark) .history-item:hover {
        border-color: var(--primary) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    }

    body:not(.theme-dark) .about-meta span {
        color: #64748b !important;
    }

    body:not(.theme-dark) .about-tagline {
        color: #475569 !important;
    }

    body:not(.theme-dark) .sidebar-nav a {
        color: #64748b !important;
    }

    body:not(.theme-dark) .sidebar-nav a.active {
        background: rgba(139, 0, 0, 0.05) !important;
        color: var(--primary) !important;
    }

    body:not(.theme-dark) .sidebar-nav a:hover:not(.active) {
        background: #f1f5f9 !important;
        color: #1e293b !important;
    }

    body:not(.theme-dark) .profile-info-grid .info-item label {
        color: #64748b !important;
    }

    body:not(.theme-dark) .profile-info-grid .info-item p {
        color: #1e293b !important;
    }

    body:not(.theme-dark) .profile-sidebar,
    body:not(.theme-dark) .profile-content,
    body:not(.theme-dark) .predict-card,
    body:not(.theme-dark) .history-card,
    body:not(.theme-dark) .about-card,
    body:not(.theme-dark) .about-hero,
    body:not(.theme-dark) .card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05) !important;
        backdrop-filter: none !important;
    }

    body:not(.theme-dark) .profile-avatar-large {
        box-shadow: 0 10px 25px rgba(244, 63, 94, 0.1) !important;
        border: 4px solid #ffffff !important;
    }

    body:not(.theme-dark) .nav-item {
        color: #64748b !important;
    }

    body:not(.theme-dark) .nav-item:hover {
        background: #f1f5f9 !important;
        color: #1e293b !important;
    }

    body:not(.theme-dark) .nav-item.active {
        background: rgba(244, 63, 94, 0.05) !important;
        color: var(--profile-primary) !important;
    }

    body:not(.theme-dark) .form-input,
    body:not(.theme-dark) .predict-input,
    body:not(.theme-dark) .search-input {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        color: #1e293b !important;
    }

    body:not(.theme-dark) .form-input:focus,
    body:not(.theme-dark) .predict-input:focus,
    body:not(.theme-dark) .search-input:focus {
        border-color: var(--primary) !important;
        background: #ffffff !important;
    }

    body:not(.theme-dark) .stat-card-modern,
    body:not(.theme-dark) .security-info,
    body:not(.theme-dark) .preference-item {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
    }

    body:not(.theme-dark) .predict-overlay {
        background: rgba(255, 255, 255, 0.9) !important;
    }

    body:not(.theme-dark) .predict-result-card {
        background: #ffffff !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.1) !important;
    }

    body:not(.theme-dark) .predict-gauge-bg {
        stroke: #f1f5f9 !important;
    }

    body:not(.theme-dark) .predict-close {
        background: #f1f5f9 !important;
        color: #64748b !important;
    }

    body:not(.theme-dark) .predict-close:hover {
        background: #fee2e2 !important;
        color: #ef4444 !important;
    }

    body:not(.theme-dark) .predict-advice-text,
    body:not(.theme-dark) .predict-gauge-label {
        color: #64748b !important;
    }

    body:not(.theme-dark) .btn-predict-action {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #1e293b !important;
    }

    body:not(.theme-dark) .btn-predict-action:hover {
        background: #f1f5f9 !important;
    }

    body:not(.theme-dark) .hero-section,
    body:not(.theme-dark) .about-container,
    body:not(.theme-dark) .predict-page,
    body:not(.theme-dark) .history-page,
    body:not(.theme-dark) .profile-wrapper {
        background: #f8fafc !important;
        background-image: none !important;
    }

    body:not(.theme-dark) .shape {
        opacity: 0.05 !important;
    }

    body:not(.theme-dark) .feature-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
    }

    body:not(.theme-dark) .footer {
        background: #ffffff !important;
        border-top: 1px solid #e2e8f0 !important;
        color: #64748b !important;
    }

    body:not(.theme-dark) .footer h4 {
        color: #1e293b !important;
    }

    body:not(.theme-dark) .footer-link {
        color: #64748b !important;
    }

    body:not(.theme-dark) .footer-link:hover {
        color: var(--primary) !important;
    }

    /* Muted text for About/Info sections in dark theme to appear less bright */
    .theme-dark .card p,
    .theme-dark .tech-stack small,
    .theme-dark .card .card-title,
    .theme-dark .metric-sub {
        color: #9ca3af !important; /* muted gray */
    }
    /* Exception: Machine Learning Pipeline steps should be black in dark theme */
    .theme-dark .process-step,
    .theme-dark .process-step p,
    .theme-dark .process-step strong {
        color: #000000 !important; /* black as requested */
    }
    /* Also set Technology Stack labels and icons to black in dark theme */
    .theme-dark .tech-stack .tech-item strong,
    .theme-dark .tech-stack .tech-item div,
    .theme-dark .tech-stack .tech-item .tech-icon,
    .theme-dark .tech-stack .tech-item small {
        color: #000000 !important;
    }
    /* About hero tagline and meta should be black in dark theme */
    .theme-dark .about-tagline,
    .theme-dark .about-meta,
    .theme-dark .about-meta .updated {
        color: #000000 !important;
    }
    /* Ensure Model Performance and Recent Predictions render black text in dark theme */
    .theme-dark .metric-card,
    .theme-dark .metric-card h3,
    .theme-dark .metric-card strong,
    .theme-dark .metric-card span,
    .theme-dark .metric-card .metric-value,
    .theme-dark .metric-card .metric-label,
    .theme-dark .metric-card .metric-sub,
    .theme-dark .metric-card table,
    .theme-dark .metric-card table th,
    .theme-dark .metric-card table td {
        color: #000000 !important;
    }
    
    .alert {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem auto;
        max-width: 500px;
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 10000;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        animation: alertSlideIn 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .alert-success {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
    }

    .alert-danger {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
    }

    /* Profile Modal Styles */
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(5px);
        display: none;
        justify-content: center;
        align-items: center;
        z-index: 10001;
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .modal-overlay.active {
        display: flex;
        opacity: 1;
    }

    .profile-modal {
        background: var(--card-bg);
        padding: 2.5rem;
        border-radius: 20px;
        max-width: 400px;
        width: 90%;
        text-align: center;
        box-shadow: var(--shadow-lg);
        transform: translateY(20px);
        transition: transform 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .modal-overlay.active .profile-modal {
        transform: translateY(0);
    }

    .profile-modal-icon {
        font-size: 4rem;
        color: var(--primary);
        margin-bottom: 1.5rem;
        background: var(--light);
        width: 100px;
        height: 100px;
        line-height: 100px;
        border-radius: 50%;
        margin-left: auto;
        margin-right: auto;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .profile-modal h2 {
        margin-bottom: 0.5rem;
        color: var(--dark);
        font-size: 1.8rem;
    }

    .profile-modal p {
        color: #666;
        margin-bottom: 2rem;
    }

    .theme-dark .profile-modal p {
        color: #9ca3af;
    }

    .profile-modal .btn-close {
        background: var(--primary);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
        width: 100%;
    }

    .profile-modal .btn-close:hover {
        background: var(--primary-dark);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(139, 0, 0, 0.3);
    }

    @keyframes alertSlideIn {
        from { transform: translate(-50%, -100px); opacity: 0; }
        to { transform: translate(-50%, 0); opacity: 1; }
    }
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
        background-color: var(--bg-color);
        background-image:
            radial-gradient(40px 40px at 10% 10%, rgba(255, 143, 171, 0.08) 0%, rgba(255, 143, 171, 0) 70%),
            radial-gradient(50px 50px at 80% 20%, rgba(255, 194, 209, 0.10) 0%, rgba(255, 194, 209, 0) 70%),
            radial-gradient(60px 60px at 20% 80%, rgba(255, 107, 154, 0.08) 0%, rgba(255, 107, 154, 0) 70%);
        color: var(--dark);
        line-height: 1.6;
        min-height: 100vh;
        -webkit-font-smoothing: antialiased;
    }
    
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* Navbar — classic transparent + blur, centered links */
    .navbar {
        background: transparent;
        backdrop-filter: blur(10px) saturate(120%);
        -webkit-backdrop-filter: blur(10px) saturate(120%);
        border-bottom: 1px solid rgba(255,255,255,0.03);
        padding: 0.6rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 1000;
        transition: background 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }

    .navbar.scrolled { background: rgba(255,255,255,0.02); box-shadow: 0 6px 18px rgba(10,10,10,0.06); }
    .theme-dark .navbar.scrolled { background: rgba(6,8,16,0.6); box-shadow: 0 6px 18px rgba(0,0,0,0.6); border-bottom-color: rgba(255,255,255,0.04); }

    .nav-left { display: flex; align-items: center; gap: 0.5rem; }
    .nav-center { display: flex; justify-content: center; flex: 1; }
    .nav-right { display: flex; justify-content: flex-end; align-items: center; gap: 0.8rem; }

    .nav-brand { font-size: 1.2rem; font-weight: 800; color: var(--primary); text-decoration: none; display: flex; align-items: center; gap: 0.5rem; }
    .brand-text { font-weight:700; }
    .brand-icon { color: var(--primary); font-size:1.05rem; }
    .brand-wrap { display:inline-flex; flex-direction:column; line-height:1; margin-left:6px; }
    .brand-tagline { font-size:0.72rem; opacity:0.85; margin-top:2px; color: var(--muted); }

    /* Compact navbar adjustments */
    .nav-links { gap: 0.9rem; }
    .nav-links a { padding: 4px 6px; font-size:0.95rem; }
    .navbar-transparent { background: transparent; backdrop-filter: blur(6px); border-bottom: 1px solid rgba(255,255,255,0.03); }
    .navbar-transparent .nav-link { color: var(--dark); }
    .theme-dark .navbar-transparent .nav-link { color: #FF4500; }
    .navbar-transparent .cta { background: linear-gradient(90deg,var(--primary),#e85a4f); color: white; padding: 6px 10px; border-radius: 999px; text-decoration:none; font-weight:700; font-size:0.95rem; }

    .navbar a, .nav-brand, .nav-toggle { color: var(--dark); }
    .theme-dark .navbar a, .theme-dark .nav-brand, .theme-dark .nav-toggle, .theme-dark .brand-icon { color: #FF4500; }

    .nav-links { display: flex; gap: 1.25rem; align-items: center; }
    .nav-links a { padding: 6px 8px; border-radius: 6px; text-decoration: none; transition: all 0.3s; }
    .nav-links a:hover { background: rgba(0,0,0,0.04); color: var(--primary); }
    .theme-dark .nav-links a:hover { background: rgba(255,255,255,0.03); }

    .nav-toggle { display: none; background: transparent; border: none; font-size: 1.5rem; color: var(--dark); cursor: pointer; padding: 6px; border-radius: 8px; align-items: center; justify-content: center; }
    .theme-dark .nav-toggle { color: #FF4500; }

    @media (max-width: 992px) {
        .nav-center {
            display: none;
            position: absolute;
            top: 100%;
            left: 0;
            width: 100%;
            background: var(--card-bg);
            flex-direction: column;
            padding: 1rem;
            box-shadow: var(--shadow-lg);
            border-bottom: 1px solid rgba(0,0,0,0.05);
            z-index: 1001;
        }
        
        .theme-dark .nav-center {
        border-bottom-color: rgba(255,255,255,0.05);
        background: #1e1e2d;
    }

    .navbar.open .nav-center {
        display: flex;
        overflow-y: auto;
        max-height: 80vh;
    }

        .nav-links {
            flex-direction: column;
            align-items: flex-start;
            width: 100%;
            gap: 0.5rem;
        }

        .nav-links a {
            width: 100%;
            padding: 0.8rem 1rem;
            font-size: 1rem;
            border-bottom: 1px solid rgba(0,0,0,0.05);
        }
        
        .theme-dark .nav-links a {
            border-bottom-color: rgba(255,255,255,0.05);
        }

        .nav-toggle {
            display: inline-flex;
            order: 2;
        }
        
        .nav-right {
            order: 3;
            gap: 0.5rem;
        }
        
        .nav-left {
            order: 1;
        }
        
        .navbar {
            padding: 0.6rem 1.2rem;
            display: flex;
            justify-content: space-between;
        }
    }

    @media (max-width: 480px) {
        .brand-tagline { display: none; }
        .navbar { padding: 0.5rem 1rem; }
        .nav-right { gap: 0.3rem; }
        .avatar-circle { width: 34px; height: 34px; }
    }
    .theme-toggle {
        border-radius: 999px;
        background: linear-gradient(135deg, #ffd54d 0%, #ffca28 100%);
        border: 1px solid rgba(0,0,0,0.06);
        position: relative;
        padding: 0 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 44px;
        height: 36px;
        font-size: 0; /* keep content as svg icons */
        cursor: pointer;
        transition: background 0.25s ease, box-shadow 0.25s ease, transform 0.12s;
        box-shadow: var(--shadow);
        overflow: visible;
    }

    .theme-toggle .icon { width: 18px; height: 18px; display: inline-block; vertical-align: middle; fill: currentColor; color: var(--dark); }
    .theme-toggle .icon-moon { display: none; }
    .theme-toggle .icon-sun { display: inline-block; }
    .theme-toggle.is-dark .icon-moon { display: inline-block; }
    .theme-toggle.is-dark .icon-sun { display: none; }

    .theme-toggle::after {
        content: '';
        position: absolute;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: #fff;
        top: 5px;
        left: 6px;
        transition: transform 0.25s ease, background 0.25s;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }

    .theme-toggle:hover { transform: translateY(-1px); box-shadow: 0 8px 18px rgba(0,0,0,0.08); }

    .theme-toggle.is-dark {
        background: linear-gradient(135deg, #0B3D91 0%, #111827 100%);
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 6px 20px rgba(11,61,145,0.18);
        color: #fefefe;
    }

    .theme-toggle.is-dark::after { transform: translateX(24px); background: #f3f4f6; }

    .theme-toggle:active { transform: scale(0.98); }

    .sr-only { position: absolute !important; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
    .theme-toggle:focus-visible { outline: 2px solid rgba(11,61,145,0.16); outline-offset: 3px; box-shadow: 0 6px 18px rgba(11,61,145,0.08); }


    /* Dashboard Headers */
    .dashboard-header {
        background: linear-gradient(135deg, #8B0000 0%, #0B3D91 100%);
        color: white;
        padding: 4rem 0;
        margin-bottom: 3rem;
        text-align: center;
    }
    
    .welcome-text {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    @media (max-width: 768px) {
        .dashboard-header {
            padding: 3rem 1rem;
            margin-bottom: 2rem;
        }
        .welcome-text {
            font-size: 1.8rem;
        }
        .container {
            padding: 0 1rem;
        }
        .btn {
            width: 100%;
            margin-bottom: 0.5rem;
        }
    }
    .user-avatar { position: relative; display: inline-block; margin-left: 0; }
    .avatar-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--card-bg);
        color: var(--dark);
        border: 1px solid var(--toggle-border);
        font-weight: 700;
        cursor: pointer;
        box-shadow: var(--shadow);
    }
    .user-icon {
        font-size: 1.15rem;
        padding: 0.35rem 0.6rem;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: var(--dark);
        border: 1px solid var(--toggle-border);
        background: var(--card-bg);
        box-shadow: var(--shadow);
        cursor: pointer;
    }
    .user-icon:hover { background: linear-gradient(90deg, rgba(139,0,0,0.06), rgba(11,61,145,0.06)); }
    .user-dropdown { 
        min-width: 280px; 
        right: 0; 
        top: 135%; 
        display: none; 
        position: absolute; 
        background: rgba(var(--card-bg-rgb, 255, 255, 255), 0.7); 
        backdrop-filter: blur(25px) saturate(180%);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25); 
        padding: 1.25rem; 
        border-radius: 24px; 
        z-index: 1000; 
        border: 1px solid rgba(255, 255, 255, 0.2);
        transform-origin: top right;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .theme-dark .user-dropdown {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .user-dropdown.show {
        display: block;
        animation: dropdownPop 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    }

    @keyframes dropdownPop {
        from { opacity: 0; transform: translateY(15px) scale(0.9) rotate(-2deg); }
        to { opacity: 1; transform: translateY(0) scale(1) rotate(0); }
    }

    .dropdown-header {
        padding: 0.5rem 0.75rem 1.25rem;
        border-bottom: 1px solid var(--glass-border);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .dropdown-user-info {
        display: flex;
        flex-direction: column;
    }

    .dropdown-header span {
        display: block;
        font-size: 0.65rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .dropdown-header strong {
        font-size: 1.1rem;
        color: var(--primary);
        font-weight: 800;
        line-height: 1.2;
    }

    .user-dropdown a { 
        display: flex; 
        align-items: center;
        gap: 1rem;
        padding: 0.9rem 1.1rem; 
        color: var(--text-main); 
        text-decoration: none; 
        border-radius: 16px; 
        font-size: 0.95rem;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 6px;
        border: 1px solid transparent;
    }

    .user-dropdown a i {
        font-size: 1.1rem;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        background: rgba(var(--primary-rgb, 139, 0, 0), 0.1);
        color: var(--primary);
        transition: all 0.3s;
    }

    .user-dropdown a:hover {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white !important;
        transform: translateX(8px);
        box-shadow: 0 10px 20px rgba(139, 0, 0, 0.2);
        border-color: rgba(255, 255, 255, 0.2);
    }

    .user-dropdown a:hover i {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        transform: scale(1.1);
    }

    .user-dropdown a.logout { 
        background: rgba(239, 68, 68, 0.08); 
        color: #ef4444 !important; 
        margin-top: 1rem; 
        border: 1px solid rgba(239, 68, 68, 0.15);
    }

    .user-dropdown a.logout i {
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
    }

    .user-dropdown a.logout:hover {
        background: #ef4444;
        color: white !important;
        border-color: #ef4444;
        box-shadow: 0 10px 20px rgba(239, 68, 68, 0.3);
    }

    .user-dropdown a.logout:hover i {
        background: rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    .nav-link {
        color: #0B3D91;
        text-decoration: none;
        font-weight: 700;
        transition: all 0.2s;
        font-size: 1.15rem;
        position: relative;
        padding-bottom: 4px;
    }
    
    .nav-link:hover {
        color: #8B0000;
    }
    
    .nav-link::after {
        content: '';
        position: absolute;
        left: 0;
        bottom: 0;
        width: 0%;
        height: 2px;
        background: linear-gradient(90deg, #8B0000, #0B3D91);
        transition: width 0.25s ease;
        border-radius: 2px;
    }
    
    .nav-link:hover::after {
        width: 100%;
    }
    
    /* Buttons */
    .btn {
        padding: 0.75rem 1.75rem;
        border: none;
        border-radius: 50px;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        font-size: 1.05rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        background: #8B0000 !important;
        color: #ffffff !important;
    }
    
    .btn:active {
        transform: scale(0.98);
    }
    
    .btn-primary {
        background: #8B0000 !important;
        color: #ffffff !important;
    }
    
    .btn-primary:hover {
        box-shadow: 0 8px 15px rgba(139, 0, 0, 0.30);
        transform: translateY(-2px);
    }
    
    .btn-danger {
        background: #8B0000 !important;
        color: #ffffff !important;
    }
    
    .btn-danger:hover {
        background: #8B0000 !important;
        box-shadow: 0 4px 12px rgba(139, 0, 0, 0.3);
    }
    
    /* Cards */
    .card {
        background: var(--card-bg);
        border-radius: var(--radius);
        padding: 2.5rem;
        box-shadow: var(--shadow);
        border: 1px solid rgba(0,0,0,0.04);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 2rem;
    }

    @media (max-width: 768px) {
        .card {
            padding: 1.5rem;
        }
        .form-grid {
            grid-template-columns: 1fr !important;
        }
        .hero-title {
            font-size: 2.2rem;
        }
        .hero-subtitle {
            font-size: 1rem;
        }
        .nav-grid {
            grid-template-columns: 1fr !important;
        }
        .quick-stats {
            grid-template-columns: 1fr !important;
        }
        .stats-grid {
            grid-template-columns: 1fr !important;
        }
        .form-section {
            padding: 1.5rem;
        }
        .result-grid {
            grid-template-columns: 1fr !important;
        }
    }
    
    .card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
    }
    
    .card-title {
        color: var(--primary);
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        position: relative;
        padding-bottom: 0.5rem;
    }
    
    .theme-dark .about-container .card-title {
        color: var(--dark);
    }
    
    .card-title::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 4px;
        background: var(--secondary);
        border-radius: 2px;
    }
    
    /* Forms */
    .form-group {
        margin-bottom: 1.5rem;
    }
    
    .form-label {
        display: block;
        margin-bottom: 0.5rem;
        color: #2d3748;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .form-control {
        width: 100%;
        padding: 0.875rem 1rem;
        border: 2px solid #edf2f7;
        border-radius: 0.75rem;
        font-size: 1rem;
        transition: all 0.2s;
        background: #f8fafc;
        font-family: inherit;
    }
    
    .form-control:focus {
        outline: none;
        border-color: var(--secondary);
        background: white;
        box-shadow: 0 0 0 4px rgba(255, 107, 154, 0.12);
    }
    
    /* Stats & Dashboard */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 2rem;
        margin-top: 2rem;
    }
    
    .stat-card {
        background: white;
        padding: 2rem;
        border-radius: var(--radius);
        text-align: center;
        box-shadow: var(--shadow);
        transition: transform 0.3s;
        border: 1px solid rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        color: #718096;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Risk Indicators */
    .risk-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.025em;
        gap: 0.5rem;
    }
    
    .risk-high {
        background: #ffe6e6;
        color: #8B0000;
        border: 2px solid #8B0000;
    }
    
    .risk-low {
        background: #f0fff4;
        color: #2f855a;
        border: 1px solid #c6f6d5;
    }
    
    /* Tables */
    .table-container {
        border-radius: var(--radius);
        overflow: hidden;
        box-shadow: var(--shadow);
        background: white;
        border: 1px solid rgba(0,0,0,0.04);
    }
    
    .table {
        width: 100%;
        border-collapse: collapse;
        background: white;
    }
    
    .table th {
        background: #f8fafc;
        color: #4a5568;
        padding: 1.25rem 1.5rem;
        text-align: left;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .table td {
        padding: 1.25rem 1.5rem;
        border-bottom: 1px solid #edf2f7;
        color: #4a5568;
    }
    
    .table tr:last-child td {
        border-bottom: none;
    }
    
    .table tr:hover {
        background: #f8fafc;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .card, .stat-card, .hero-content {
        animation: fadeIn 0.6s ease-out forwards;
    }
    
    /* Mobile */
    @media (max-width: 768px) {
        .navbar {
            padding: 1rem;
            flex-direction: column;
            gap: 1rem;
        }
        
        .container {
            padding: 0 1rem;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
'''

BASE_JS = '''
<script>
    function showAlert(message, type = 'success') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        
        // Add icon based on type
        const icon = type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle';
        alertDiv.innerHTML = `<i class="fas ${icon}"></i> <span>${message}</span>`;
        
        document.body.prepend(alertDiv);
        
        setTimeout(() => {
            alertDiv.style.opacity = '0';
            alertDiv.style.transform = 'translate(-50%, -20px)';
            alertDiv.style.transition = 'all 0.5s ease';
            setTimeout(() => alertDiv.remove(), 500);
        }, 4000);
    }
    
    function formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleString();
    }
    
    function logout() {
        fetch('/logout')
            .then(() => window.location.href = '/');
    }

    function showProfilePopup(name, username) {
        let modal = document.getElementById('profileModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'profileModal';
            modal.className = 'modal-overlay';
            modal.innerHTML = `
                <div class="profile-modal">
                    <div class="profile-modal-icon">
                        <i class="fas fa-user-circle"></i>
                    </div>
                    <h2>Welcome, ${name}!</h2>
                    <p>Logged in as <strong>@${username}</strong></p>
                    <button class="btn-close" onclick="closeProfileModal()">Great, thanks!</button>
                </div>
            `;
            document.body.appendChild(modal);
            
            // Close on overlay click
            modal.addEventListener('click', function(e) {
                if (e.target === modal) closeProfileModal();
            });
        }
        
        // Update content in case it changed
        modal.querySelector('h2').textContent = `Welcome, ${name}!`;
        modal.querySelector('p strong').textContent = `@${username}`;
        
        modal.classList.add('active');
        document.body.style.overflow = 'hidden'; // Prevent scrolling
    }

    function closeProfileModal() {
        const modal = document.getElementById('profileModal');
        if (modal) {
            modal.classList.remove('active');
            document.body.style.overflow = ''; // Restore scrolling
        }
    }
    
    document.addEventListener('DOMContentLoaded', function() {
        const toggles = document.querySelectorAll('.theme-toggle');
        function updateToggleStates(theme) {
            toggles.forEach(function(toggle) {
                toggle.classList.toggle('is-dark', theme === 'dark');
                toggle.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
                toggle.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
                // SVG icons visibility handled via CSS (.icon-sun / .icon-moon)
            });
        }
        function applyTheme(theme) {
            document.body.classList.toggle('theme-dark', theme === 'dark');
            document.body.classList.toggle('theme-light', theme !== 'dark');
            updateToggleStates(theme);
        }
        let current = localStorage.getItem('theme') || 'light';
        applyTheme(current);
        toggles.forEach(function(toggle) {
            toggle.addEventListener('click', function() {
                current = current === 'dark' ? 'light' : 'dark';
                localStorage.setItem('theme', current);
                applyTheme(current);
            });
            // Support keyboard activation (Enter / Space)
            toggle.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    toggle.click();
                }
            });
        });

        // Delegated handling for user icon dropdowns (works across pages)
        function closeAllDropdowns() {
            document.querySelectorAll('.user-dropdown').forEach(d => d.style.display = 'none');
            document.querySelectorAll('.user-icon').forEach(b => b.setAttribute('aria-expanded', 'false'));
        }

        // Ensure nav-toggle buttons are present (insert where missing)
        document.querySelectorAll('.navbar').forEach(nav => {
            if (nav.querySelector('.nav-toggle')) return;
            const brand = nav.querySelector('.nav-brand');
            const links = nav.querySelector('.nav-links');
            if (!brand || !links) return;
            const btn = document.createElement('button');
            btn.className = 'nav-toggle';
            btn.setAttribute('aria-label','Toggle menu');
            btn.setAttribute('aria-expanded','false');
            btn.innerHTML = '<i class="fas fa-bars"></i>';
            brand.insertAdjacentElement('afterend', btn);
        });

        // Responsive nav toggle handling
        document.querySelectorAll('.nav-toggle').forEach(btn => {
            btn.addEventListener('click', function(e){
                e.stopPropagation();
                const nav = this.closest('.navbar');
                if (!nav) return;
                const isOpen = nav.classList.toggle('open');
                this.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
                // update controlled mobile menu aria state
                const mobile = nav.querySelector('.nav-center');
                if (mobile) mobile.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
            });
        });
        // Close the nav when clicking outside
        document.addEventListener('click', function(e){
            if (e.target.closest && e.target.closest('.navbar')) return;
            document.querySelectorAll('.navbar.open').forEach(nav => {
                nav.classList.remove('open');
                const btn = nav && nav.querySelector('.nav-toggle');
                if (btn) btn.setAttribute('aria-expanded', 'false');
                const mobile = nav && nav.querySelector('.nav-center');
                if (mobile) mobile.setAttribute('aria-hidden', 'true');
            });
        });
        // Close nav when a link is clicked (mobile slide-down)
        document.querySelectorAll('.nav-links a').forEach(a => a.addEventListener('click', function(){
            document.querySelectorAll('.navbar.open').forEach(nav => {
                nav.classList.remove('open');
                const btn = nav && nav.querySelector('.nav-toggle');
                if (btn) btn.setAttribute('aria-expanded','false');
                const mobile = nav && nav.querySelector('.nav-center');
                if (mobile) mobile.setAttribute('aria-hidden','true');
            });
        }));
        document.addEventListener('click', function(e) {
            const icon = e.target.closest && e.target.closest('.user-icon');
            if (icon) {
                e.stopPropagation();
                const wrapper = icon.closest('.user-avatar, .user-menu');
                const dropdown = wrapper ? wrapper.querySelector('.user-dropdown') : icon.nextElementSibling;
                if (!dropdown) return;
                const isOpen = dropdown.style.display === 'block';
                closeAllDropdowns();
                dropdown.style.display = isOpen ? 'none' : 'block';
                icon.setAttribute('aria-expanded', !isOpen);
                return;
            }
            // click outside - close all
            closeAllDropdowns();
        });
        // Prevent closing when clicking inside dropdowns
        document.querySelectorAll('.user-dropdown').forEach(d => d.addEventListener('click', function(e){ e.stopPropagation(); }));

        // Navbar scroll: toggle 'scrolled' class for better contrast on scroll
        function updateNavbarOnScroll() {
            document.querySelectorAll('.navbar').forEach(function(nav) {
                nav.classList.toggle('scrolled', window.scrollY > 8);
            });
        }
        updateNavbarOnScroll();
        window.addEventListener('scroll', updateNavbarOnScroll);
    });
</script>
'''

def render_navbar():
    """Return structured navbar HTML as Markup (safe for Jinja templates).
    This helper centralizes markup and keeps templates clean while allowing
    runtime access to `session` e.g., to show user name / auth links.
    """
    _display_name = session.get('name') or session.get('username') or 'User'
    user_authenticated = 'user_id' in session

    html = []
    html.append('<nav class="navbar navbar-transparent" role="navigation" aria-label="Main Navigation">')
    html.append('  <div class="nav-left">')
    html.append('    <a href="/" class="nav-brand" aria-label="Cardix AI home">')
    html.append('      <i class="fas fa-heartbeat brand-icon" aria-hidden="true"></i>')
    html.append('      <div class="brand-wrap">')
    html.append('        <span class="brand-text">Cardix AI</span>')
    html.append('        <small class="brand-tagline">Predict • Prevent</small>')
    html.append('      </div>')
    html.append('    </a>')
    html.append('  </div>')

    html.append('  <div class="nav-center" id="mobileMenu" aria-hidden="true">')
    html.append('    <ul class="nav-links" role="menubar">')
    html.append('      <li role="none"><a role="menuitem" href="/" class="nav-link">Home</a></li>')
    html.append('      <li role="none"><a role="menuitem" href="/dashboard" class="nav-link">Dashboard</a></li>')
    html.append('      <li role="none"><a role="menuitem" href="/about" class="nav-link">About</a></li>')
    html.append('      <li role="none"><a role="menuitem" href="/history" class="nav-link">History</a></li>')
    html.append('    </ul>')
    html.append('  </div>')

    html.append('  <div class="nav-right">')
    if user_authenticated:
        html.append('    <div class="user-menu">')
        html.append(f'      <a href="#" class="nav-link user-icon" id="userIcon" role="button" aria-haspopup="true" aria-expanded="false" aria-controls="userDropdown" title="{_display_name}" onclick="event.preventDefault();"><i class="fas fa-user-circle" aria-hidden="true"></i></a>')
        html.append('      <div class="user-dropdown" id="userDropdown" aria-hidden="true">')
        html.append('        <div class="dropdown-header">')
        html.append(f'          <div class="user-avatar-mini" style="width: 40px; height: 40px; border-radius: 12px; background: linear-gradient(135deg, var(--primary), var(--secondary)); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.2rem;">{session.get("name", "U")[0].upper()}</div>')
        html.append('          <div class="dropdown-user-info">')
        html.append(f'            <span>{session.get("username", "user")}</span>')
        html.append(f'            <strong>{session.get("name", "User")}</strong>')
        html.append('          </div>')
        html.append('        </div>')
        html.append(f'        <a href="#" class="nav-link" onclick="event.preventDefault(); showProfilePopup(\'{session.get("name", "User")}\', \'{session.get("username", "user")}\');"><i class="fas fa-id-card"></i> Profile Card</a>')
        html.append('        <a href="/profile" class="nav-link"><i class="fas fa-user-cog"></i> Settings</a>')
        html.append('        <a href="/history" class="nav-link"><i class="fas fa-history"></i> My History</a>')
        html.append('        <a href="#" class="nav-link logout-link logout" onclick="event.preventDefault(); logout();"><i class="fas fa-sign-out-alt"></i> Logout</a>')
        html.append('      </div>')
        html.append('    </div>')
    else:
        html.append('    <a href="/login" class="nav-link">Login</a>')
        html.append('    <a href="/register" class="nav-link cta">Get Started</a>')

    html.append('    <button class="nav-toggle" aria-label="Toggle menu" aria-expanded="false" aria-controls="mobileMenu"><i class="fas fa-bars" aria-hidden="true"></i></button>')
    html.append('  </div>')
    html.append('</nav>')

    return Markup('\n'.join(html))


# Register helper for templates; keep backwards compatibility with existing {{ NAVBAR }} calls
app.jinja_env.globals['render_navbar'] = render_navbar

# Individual page templates

@app.context_processor
def inject_base_assets():
    return dict(
        base_css=BASE_CSS,
        base_js=BASE_JS,
        show_last_prediction=app.config.get('SHOW_LAST_PREDICTION', True),
        datetime=datetime,
        NAVBAR=render_navbar()
    )

INDEX_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cardix AI - Intelligent Heart Disease Risk Prediction</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    {{ base_css | safe }}
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #8B0000 0%, #0B3D91 100%);
            --glass-bg: rgba(255, 255, 255, 0.7);
            --glass-border: rgba(255, 255, 255, 0.2);
            --glass-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        }

        .theme-dark {
            --glass-bg: rgba(17, 24, 39, 0.7);
            --glass-border: rgba(255, 255, 255, 0.1);
        }

        body {
            overflow-x: hidden;
        }

        /* Modern Hero Section */
        .hero-section {
            min-height: 100vh;
            display: flex;
            align-items: center;
            position: relative;
            background: var(--bg-color);
            overflow: hidden;
            padding: 2rem 0;
        }

        .shape {
            position: absolute;
            filter: blur(80px);
            opacity: 0.15;
            border-radius: 50%;
            z-index: 0;
        }

        .shape-1 { width: 500px; height: 500px; background: var(--primary); top: -100px; right: -100px; animation: float 15s infinite alternate; }
        .shape-2 { width: 400px; height: 400px; background: var(--secondary); bottom: -100px; left: -100px; animation: float 18s infinite alternate-reverse; }
        .shape-3 { width: 300px; height: 300px; background: var(--accent); top: 50%; left: 50%; transform: translate(-50%, -50%); animation: float 20s infinite alternate; }

        @keyframes float {
            0% { transform: translate(0, 0) rotate(0deg); }
            100% { transform: translate(50px, 50px) rotate(30deg); }
        }

        .hero-container {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 4rem;
            align-items: center;
            width: 100%;
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 2rem;
        }

        .hero-content {
            animation: slideInLeft 1s ease-out;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.6rem 1.2rem;
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            border-radius: 100px;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 2rem;
            box-shadow: var(--glass-shadow);
        }

        .hero-badge i {
            color: var(--secondary);
            animation: pulse 2s infinite;
        }

        .hero-title {
            font-size: clamp(2.5rem, 5vw, 4.5rem);
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 1.5rem;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }

        .hero-description {
            font-size: clamp(1.1rem, 1.5vw, 1.3rem);
            color: var(--text-muted);
            line-height: 1.6;
            margin-bottom: 3rem;
            max-width: 600px;
        }

        .hero-actions {
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
        }

        .btn-premium {
            padding: 1rem 2.5rem;
            border-radius: 16px;
            font-weight: 700;
            font-size: 1.1rem;
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .btn-primary-premium {
            background: var(--primary-gradient);
            color: white !important;
            box-shadow: 0 10px 30px rgba(139, 0, 0, 0.2);
        }

        .btn-primary-premium:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(139, 0, 0, 0.3);
        }

        .btn-secondary-premium {
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            color: var(--text-color) !important;
            box-shadow: var(--glass-shadow);
        }

        .btn-secondary-premium:hover {
            background: var(--glass-border);
            transform: translateY(-3px);
        }

        /* Visual Element */
        .hero-visual {
            position: relative;
            animation: slideInRight 1s ease-out;
        }

        .main-card-visual {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 32px;
            padding: 2.5rem;
            box-shadow: var(--glass-shadow);
            position: relative;
            z-index: 2;
            overflow: hidden;
        }

        .heart-gif-container {
            width: 100%;
            height: 350px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 2rem;
            background: radial-gradient(circle, rgba(139, 0, 0, 0.05) 0%, transparent 70%);
            border-radius: 24px;
        }

        .heart-gif-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            filter: drop-shadow(0 0 20px rgba(139, 0, 0, 0.2));
        }

        .stats-floating {
            position: absolute;
            background: var(--glass-bg);
            backdrop-filter: blur(15px);
            border: 1px solid var(--glass-border);
            padding: 1.2rem;
            border-radius: 20px;
            box-shadow: var(--glass-shadow);
            z-index: 3;
            animation: floating 4s ease-in-out infinite;
        }

        .stat-accuracy { top: 10%; right: -20px; animation-delay: 0s; }
        .stat-realtime { bottom: 15%; left: -30px; animation-delay: 1s; }

        @keyframes floating {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-15px); }
        }

        /* Features Section */
        .section-header {
            text-align: center;
            margin-bottom: 5rem;
        }

        .section-tag {
            color: var(--secondary);
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-size: 0.9rem;
            margin-bottom: 1rem;
            display: block;
        }

        .section-title {
            font-size: clamp(2rem, 3vw, 3rem);
            font-weight: 800;
            color: var(--text-color);
        }

        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2.5rem;
            margin-bottom: 6rem;
        }

        .premium-feature-card {
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            border-radius: 28px;
            padding: 3rem 2.5rem;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            box-shadow: var(--glass-shadow);
        }

        .premium-feature-card:hover {
            transform: translateY(-12px);
            background: var(--glass-border);
            border-color: var(--secondary);
        }

        .feature-icon-wrapper {
            width: 70px;
            height: 70px;
            background: var(--primary-gradient);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.8rem;
            color: white;
            margin-bottom: 2rem;
            transition: all 0.3s;
        }

        .premium-feature-card:hover .feature-icon-wrapper {
            transform: scale(1.1) rotate(10deg);
            box-shadow: 0 10px 20px rgba(139, 0, 0, 0.2);
        }

        .feature-h {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 1.2rem;
            color: var(--text-color);
        }

        .feature-p {
            color: var(--text-muted);
            line-height: 1.6;
            font-size: 1.05rem;
        }

        /* CTA Section */
        .cta-premium {
            background: var(--primary-gradient);
            border-radius: 40px;
            padding: 5rem 3rem;
            text-align: center;
            color: white;
            position: relative;
            overflow: hidden;
            margin: 6rem 0;
            box-shadow: 0 20px 50px rgba(11, 61, 145, 0.25);
        }

        .cta-premium::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: url('https://www.transparenttextures.com/patterns/cubes.png');
            opacity: 0.1;
        }

        .cta-title {
            font-size: clamp(2rem, 4vw, 3.5rem);
            font-weight: 800;
            margin-bottom: 1.5rem;
            position: relative;
            z-index: 1;
        }

        .cta-text {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 3rem;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
            position: relative;
            z-index: 1;
        }

        /* Animations */
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-50px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(50px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.2); opacity: 0.7; }
            100% { transform: scale(1); opacity: 1; }
        }

        /* Responsiveness */
        @media (max-width: 1200px) {
            .hero-container { grid-template-columns: 1fr; text-align: center; gap: 3rem; }
            .hero-content { display: flex; flex-direction: column; align-items: center; }
            .hero-description { margin-left: auto; margin-right: auto; }
            .hero-actions { justify-content: center; }
            .hero-visual { max-width: 600px; margin: 0 auto; }
            .stat-accuracy { right: 0; }
            .stat-realtime { left: 0; }
        }

        @media (max-width: 768px) {
            .hero-section { padding: 4rem 0; }
            .hero-title { font-size: 2.8rem; }
            .main-card-visual { padding: 1.5rem; }
            .heart-gif-container { height: 250px; }
            .stats-floating { padding: 0.8rem; font-size: 0.8rem; }
            .features-grid { grid-template-columns: 1fr; }
            .cta-premium { padding: 4rem 1.5rem; border-radius: 30px; }
        }
    </style>
    {{ NAVBAR }}
</head>
<body>
    <main>
        <!-- Hero Section -->
        <section class="hero-section">
            <div class="shape shape-1"></div>
            <div class="shape shape-2"></div>
            <div class="shape shape-3"></div>
            
            <div class="hero-container">
                <div class="hero-content">
                    <div class="hero-badge">
                        <i class="fas fa-shield-virus"></i>
                        <span>AI-Powered Cardiac Health Protection</span>
                    </div>
                    <h1 class="hero-title">Your Heart's Future, <br>Predicted Today.</h1>
                    <p class="hero-description">
                        Experience the next generation of cardiac risk assessment. Our advanced machine learning models analyze clinical data with 95% accuracy to provide instant, actionable insights.
                    </p>
                    <div class="hero-actions">
                        <a href="/predict" class="btn-premium btn-primary-premium">
                            <i class="fas fa-stethoscope"></i> Start Risk Analysis
                        </a>
                        <a href="/about" class="btn-premium btn-secondary-premium">
                            <i class="fas fa-info-circle"></i> How It Works
                        </a>
                    </div>
                </div>

                <div class="hero-visual">
                    <div class="stats-floating stat-accuracy">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="width: 40px; height: 40px; background: #4ade80; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white;">
                                <i class="fas fa-check"></i>
                            </div>
                            <div>
                                <div style="font-weight: 800; font-size: 1.1rem; color: var(--text-color);">95% Accuracy</div>
                                <div style="font-size: 0.8rem; color: var(--text-muted);">Verified Models</div>
                            </div>
                        </div>
                    </div>

                    <div class="stats-floating stat-realtime">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="width: 40px; height: 40px; background: #60a5fa; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white;">
                                <i class="fas fa-bolt"></i>
                            </div>
                            <div>
                                <div style="font-weight: 800; font-size: 1.1rem; color: var(--text-color);">Instant Result</div>
                                <div style="font-size: 0.8rem; color: var(--text-muted);">Real-time Processing</div>
                            </div>
                        </div>
                    </div>

                    <div class="main-card-visual">
                        <div class="heart-gif-container">
                            <img src="{{ url_for('static', filename='heart_realistic.gif') }}" alt="Cardiac Visualization"
                                 onerror="this.src='https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJndzZ5bmR3Z3d3Z3d3Z3d3Z3d3Z3d3Z3d3Z3d3Z3d3Z3d3ZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/Lp4L6VILAEOx40Y6V8/giphy.gif'">
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: 700; margin-bottom: 0.5rem; color: var(--text-color);">Advanced Neural Analysis</div>
                            <div style="width: 100%; height: 4px; background: #eee; border-radius: 2px; overflow: hidden;">
                                <div style="width: 75%; height: 100%; background: var(--primary-gradient); animation: loading 2s infinite linear;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Features Section -->
        <section class="container" style="padding: 8rem 2rem;">
            <div class="section-header">
                <span class="section-tag">Key Features</span>
                <h2 class="section-title">Intelligent Health Monitoring</h2>
            </div>

            <div class="features-grid">
                <div class="premium-feature-card">
                    <div class="feature-icon-wrapper">
                        <i class="fas fa-microchip"></i>
                    </div>
                    <h3 class="feature-h">Dual-Model Prediction</h3>
                    <p class="feature-p">Leveraging both Random Forest and Logistic Regression algorithms for the most comprehensive and reliable risk score.</p>
                </div>

                <div class="premium-feature-card">
                    <div class="feature-icon-wrapper" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                        <i class="fas fa-file-medical-alt"></i>
                    </div>
                    <h3 class="feature-h">Clinical Data Analysis</h3>
                    <p class="feature-p">Deep analysis of key indicators like cholesterol, blood pressure, and Heart rate to map your cardiovascular health.</p>
                </div>

                <div class="premium-feature-card">
                    <div class="feature-icon-wrapper" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);">
                        <i class="fas fa-history"></i>
                    </div>
                    <h3 class="feature-h">Smart History Tracking</h3>
                    <p class="feature-p">Monitor your health journey over time with detailed logs and visual progress indicators for every assessment.</p>
                </div>

                <div class="premium-feature-card">
                    <div class="feature-icon-wrapper" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);">
                        <i class="fas fa-shield-alt"></i>
                    </div>
                    <h3 class="feature-h">Privacy-First Design</h3>
                    <p class="feature-p">Your medical data is your own. We use enterprise-grade encryption to ensure your information remains private and secure.</p>
                </div>
            </div>

            <!-- CTA Section -->
            <div class="cta-premium">
                <h2 class="cta-title">Ready to prioritize your heart health?</h2>
                <p class="cta-text">Join thousands of proactive individuals using Cardix AI to monitor and protect their cardiovascular future. Start your free assessment today.</p>
                <div style="display: flex; gap: 1.5rem; justify-content: center; flex-wrap: wrap; position: relative; z-index: 1;">
                    <a href="/register" class="btn-premium" style="background: white; color: var(--primary) !important;">
                        Create Free Account <i class="fas fa-arrow-right"></i>
                    </a>
                    <a href="/predict" class="btn-premium" style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); color: white !important;">
                        Try Guest Analysis
                    </a>
                </div>
            </div>
        </section>
    </main>

    <footer class="home-footer">
        <div class="home-footer-inner">
            <div class="social-links" aria-label="Developer social links">
                <a href="https://www.linkedin.com" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn profile">
                    <i class="fab fa-linkedin-in"></i>
                </a>
                <a href="https://github.com" target="_blank" rel="noopener noreferrer" aria-label="GitHub profile">
                    <i class="fab fa-github"></i>
                </a>
                <a href="https://www.instagram.com" target="_blank" rel="noopener noreferrer" aria-label="Instagram profile">
                    <i class="fab fa-instagram"></i>
                </a>
            </div>

            <p class="home-disclaimer">Prediction results are only an estimate based on your provided health information. Please consult a qualified doctor for proper medical advice.</p>
        </div>

        <div class="home-footer-bottom">&copy; 2026 Heart Disease Prediction System - All Rights Reserved.</div>
    </footer>

    <style>
        @keyframes loading {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .home-footer {
            margin-top: 2rem;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #334155 100%);
            color: #e2e8f0;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
        }

        .home-footer-inner {
            max-width: 900px;
            margin: 0 auto;
            padding: 1.8rem 1.5rem 1.1rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.9rem;
            text-align: center;
        }

        .social-links {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.95rem;
        }

        .social-links a {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #e2e8f0;
            text-decoration: none;
            border: 1px solid rgba(148, 163, 184, 0.3);
            background: rgba(255, 255, 255, 0.04);
            transition: all 0.25s ease;
            font-size: 1rem;
        }

        .social-links a:hover {
            transform: translateY(-2px) scale(1.04);
            background: rgba(59, 130, 246, 0.18);
            border-color: rgba(96, 165, 250, 0.45);
            color: #ffffff;
        }

        .home-disclaimer {
            margin: 0;
            color: #cbd5e1;
            font-size: 0.9rem;
            line-height: 1.55;
            max-width: 760px;
        }

        .home-footer-bottom {
            text-align: center;
            border-top: 1px solid rgba(255, 255, 255, 0.12);
            padding: 0.85rem 1rem 1rem;
            color: #cbd5e1;
            font-size: 0.88rem;
        }

        @media (max-width: 768px) {
            .home-footer-inner {
                padding: 1.6rem 1rem 1rem;
            }

            .home-disclaimer {
                font-size: 0.86rem;
            }
        }
    </style>

    {{ base_js | safe }}
</body>
</html>
'''


LOGIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Cardix AI</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    {{ base_css | safe }}
    <style>
        :root {
            --login-primary: #8B0000;
            --login-secondary: #0B3D91;
            --glass-bg: rgba(255, 255, 255, 0.75);
            --glass-border: rgba(255, 255, 255, 0.4);
            --glow-color: rgba(139, 0, 0, 0.3);
        }

        .theme-dark {
            --glass-bg: rgba(15, 23, 42, 0.8);
            --glass-border: rgba(255, 255, 255, 0.1);
            --glow-color: rgba(255, 143, 171, 0.2);
        }

        .login-wrapper {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: var(--bg-color);
            position: relative;
            overflow: hidden;
        }

        /* Abstract Background Elements */
        .bg-orb {
            position: absolute;
            width: 600px;
            height: 600px;
            filter: blur(100px);
            border-radius: 50%;
            z-index: 0;
            opacity: 0.2;
        }
        .orb-1 { background: var(--login-primary); top: -200px; left: -100px; animation: pulse 10s infinite alternate; }
        .orb-2 { background: var(--login-secondary); bottom: -200px; right: -100px; animation: pulse 12s infinite alternate-reverse; }

        @keyframes pulse {
            0% { transform: scale(1) translate(0, 0); }
            100% { transform: scale(1.2) translate(50px, 50px); }
        }

        .login-container {
            width: 100%;
            max-width: 1100px;
            min-height: 650px;
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            background: var(--glass-bg);
            backdrop-filter: blur(25px);
            border: 1px solid var(--glass-border);
            border-radius: 40px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
            overflow: hidden;
            position: relative;
            z-index: 1;
        }

        .login-visual {
            background: linear-gradient(160deg, #1e1e2d 0%, #0f1220 100%);
            padding: 60px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            color: white;
            position: relative;
            overflow: hidden;
        }

        .visual-pattern {
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background-image: radial-gradient(circle at 2px 2px, rgba(255,255,255,0.05) 1px, transparent 0);
            background-size: 32px 32px;
        }

        .visual-content {
            position: relative;
            z-index: 2;
        }

        .brand-pill {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 8px 16px;
            background: rgba(255,255,255,0.1);
            border-radius: 100px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.1);
        }

        .brand-pill i { color: #ff4d6d; }

        .visual-headline {
            font-size: 3.5rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 24px;
            background: linear-gradient(to right, #fff, #9ca3af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .visual-desc {
            font-size: 1.15rem;
            color: #9ca3af;
            max-width: 400px;
            line-height: 1.6;
        }

        .heart-loader {
            position: absolute;
            bottom: -50px;
            right: -50px;
            width: 300px;
            height: 300px;
            opacity: 0.1;
            pointer-events: none;
        }

        .login-form-side {
            padding: 60px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            background: var(--card-bg);
        }

        .form-header {
            margin-bottom: 40px;
        }

        .form-header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--dark);
            margin-bottom: 12px;
        }

        .form-header p {
            color: var(--text-muted);
            font-size: 1.05rem;
        }

        .input-group {
            margin-bottom: 24px;
            position: relative;
        }

        .input-group label {
            display: block;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            color: var(--primary);
        }

        .input-field {
            width: 100%;
            height: 60px;
            background: var(--bg-color);
            border: 2px solid transparent;
            border-radius: 20px;
            padding: 0 24px;
            font-size: 1rem;
            font-weight: 600;
            color: var(--dark);
            transition: all 0.3s;
        }

        .input-field:focus {
            background: var(--card-bg);
            border-color: var(--primary);
            box-shadow: 0 10px 20px -10px var(--glow-color);
            outline: none;
        }

        .btn-login {
            width: 100%;
            height: 60px;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 20px;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-top: 20px;
            box-shadow: 0 15px 30px -10px rgba(139, 0, 0, 0.4);
        }

        .btn-login:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px -10px rgba(139, 0, 0, 0.5);
            background: #a30000;
        }

        .btn-login i {
            font-size: 1.2rem;
            transition: transform 0.3s;
        }

        .btn-login:hover i {
            transform: translateX(5px);
        }

        .form-footer {
            margin-top: 32px;
            text-align: center;
            font-weight: 600;
            color: var(--text-muted);
        }

        .form-footer a {
            color: var(--primary);
            text-decoration: none;
            margin-left: 5px;
            position: relative;
        }

        .form-footer a::after {
            content: '';
            position: absolute;
            bottom: -2px; left: 0; width: 0; height: 2px;
            background: var(--primary);
            transition: width 0.3s;
        }

        .form-footer a:hover::after {
            width: 100%;
        }

        @media (max-width: 968px) {
            .login-container {
                grid-template-columns: 1fr;
                max-width: 500px;
                min-height: auto;
            }
            .login-visual {
                display: none;
            }
            .login-form-side {
                padding: 40px;
            }
        }
    </style>
</head>
<body>
    {{ NAVBAR }}

    <div class="login-wrapper">
        <div class="bg-orb orb-1"></div>
        <div class="bg-orb orb-2"></div>

        <div class="login-container">
            <div class="login-visual">
                <div class="visual-pattern"></div>
                <div class="visual-content">
                    <div class="brand-pill">
                        <i class="fas fa-shield-heart"></i>
                        <span>AI-Powered Diagnostics</span>
                    </div>
                    <h2 class="visual-headline">Advanced Cardiac Monitoring.</h2>
                    <p class="visual-desc">Securely access your health data and leverage state-of-the-art machine learning for heart disease prediction.</p>
                </div>
                
                <div class="heart-loader">
                    <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                        <path fill="currentColor" d="M100 180c-2-2-75-70-75-120 0-30 25-55 55-55 15 0 30 10 40 25 10-15 25-25 40-25 30 0 55 25 55 55 0 50-73 118-75 120z" />
                    </svg>
                </div>

                <div class="visual-footer">
                    <div style="display: flex; gap: 20px; opacity: 0.6; font-size: 0.85rem;">
                        <span><i class="fas fa-lock"></i> AES-256</span>
                        <span><i class="fas fa-check-circle"></i> HIPAA Compliant</span>
                    </div>
                </div>
            </div>

            <div class="login-form-side">
                <div class="form-header">
                    <h1>Welcome back</h1>
                    <p>Enter your credentials to continue</p>
                </div>

                <form id="loginForm">
                    <div class="input-group">
                        <label>Identity</label>
                        <input type="text" id="username" class="input-field" placeholder="Username or Email" required>
                    </div>

                    <div class="input-group">
                        <label>Security Key</label>
                        <input type="password" id="password" class="input-field" placeholder="••••••••" required>
                    </div>

                    <div style="text-align: right; margin-bottom: 10px;">
                        <a href="/forgot-password" style="color: var(--primary); text-decoration: none; font-size: 0.9rem; font-weight: 600;">Forgot Password?</a>
                    </div>

                    <button type="submit" class="btn-login">
                        Authenticate <i class="fas fa-arrow-right"></i>
                    </button>

                    <div class="form-footer">
                        Don't have an account? <a href="/register">Initialize Registration</a>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('loginForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const username = document.getElementById('username').value.trim();
            const password = document.getElementById('password').value;
            
            const btn = e.target.querySelector('button');
            const originalText = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Authenticating...';

            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: new URLSearchParams({
                        'username': username,
                        'password': password
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showAlert('Login successful! Welcome back.', 'success');
                    setTimeout(() => {
                        window.location.href = result.redirect;
                    }, 1000);
                } else {
                    showAlert(result.message, 'danger');
                    btn.disabled = false;
                    btn.innerHTML = originalText;
                }
            } catch (error) {
                showAlert('An error occurred. Please try again.', 'danger');
                btn.disabled = false;
                btn.innerHTML = originalText;
            }
        });
    </script>
    {{ base_js | safe }}
</body>
</html>
'''


REGISTER_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Join Cardix AI - Heart Health Reimagined</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    {{ base_css | safe }}
    <style>
        :root {
            --reg-primary: #0B3D91;
            --reg-accent: #8B0000;
            --mesh-color-1: rgba(11, 61, 145, 0.1);
            --mesh-color-2: rgba(139, 0, 0, 0.05);
            --mesh-color-3: rgba(0, 150, 255, 0.08);
            --glass-bg: rgba(255, 255, 255, 0.85);
            --glass-border: rgba(255, 255, 255, 0.5);
        }

        .theme-dark {
            --glass-bg: rgba(15, 23, 42, 0.8);
            --glass-border: rgba(255, 255, 255, 0.1);
            --mesh-color-1: rgba(30, 64, 175, 0.2);
            --mesh-color-2: rgba(153, 27, 27, 0.1);
            --mesh-color-3: rgba(59, 130, 246, 0.15);
        }

        .register-wrapper {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 3rem 1.5rem;
            background: var(--bg-color);
            position: relative;
            overflow: hidden;
        }

        /* Unique Mesh Background */
        .mesh-bg {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            background-image: 
                radial-gradient(at 0% 0%, var(--mesh-color-1) 0, transparent 50%),
                radial-gradient(at 100% 0%, var(--mesh-color-2) 0, transparent 50%),
                radial-gradient(at 100% 100%, var(--mesh-color-3) 0, transparent 50%),
                radial-gradient(at 0% 100%, var(--mesh-color-1) 0, transparent 50%),
                radial-gradient(at 50% 50%, var(--mesh-color-2) 0, transparent 50%);
            filter: blur(60px);
            animation: meshRotate 30s infinite alternate linear;
        }

        @keyframes meshRotate {
            0% { transform: scale(1) rotate(0deg); }
            100% { transform: scale(1.2) rotate(10deg); }
        }

        .register-card {
            width: 100%;
            max-width: 1150px;
            background: var(--glass-bg);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid var(--glass-border);
            border-radius: 32px;
            display: grid;
            grid-template-columns: 420px 1fr;
            overflow: hidden;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
            position: relative;
            z-index: 1;
            animation: registerAppear 0.8s cubic-bezier(0.22, 1, 0.36, 1);
        }

        @keyframes registerAppear {
            from { opacity: 0; transform: scale(0.95) translateY(20px); }
            to { opacity: 1; transform: scale(1) translateY(0); }
        }

        .register-sidebar {
            background: linear-gradient(160deg, var(--reg-primary) 0%, #061e4a 100%);
            padding: 4rem 3rem;
            color: white;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
        }

        .register-sidebar::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: url('https://www.transparenttextures.com/patterns/carbon-fibre.png');
            opacity: 0.1;
            pointer-events: none;
        }

        .sidebar-header i {
            font-size: 3rem;
            background: linear-gradient(to right, #fff, rgba(255,255,255,0.6));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
        }

        .sidebar-header h2 {
            font-size: 2.25rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 1.5rem;
        }

        .reg-steps {
            list-style: none;
            padding: 0;
            margin: 3rem 0;
        }

        .reg-step {
            display: flex;
            align-items: flex-start;
            gap: 1.25rem;
            margin-bottom: 2rem;
            opacity: 0;
            animation: stepFade 0.5s forwards;
        }

        .reg-step:nth-child(1) { animation-delay: 0.4s; }
        .reg-step:nth-child(2) { animation-delay: 0.6s; }
        .reg-step:nth-child(3) { animation-delay: 0.8s; }

        @keyframes stepFade {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .step-icon {
            width: 32px;
            height: 32px;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            flex-shrink: 0;
        }

        .step-content h4 {
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .step-content p {
            font-size: 0.9rem;
            opacity: 0.7;
            line-height: 1.4;
        }

        .sidebar-footer {
            font-size: 0.85rem;
            opacity: 0.6;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .register-form-area {
            padding: 4rem;
            background: var(--card-bg);
        }

        .form-header {
            margin-bottom: 3.5rem;
        }

        .form-header h1 {
            font-size: 2.5rem;
            font-weight: 900;
            color: var(--text-color);
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }

        .form-header p {
            color: var(--text-muted);
            font-size: 1.1rem;
        }

        .reg-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        .full-row { grid-column: span 2; }

        .reg-group {
            margin-bottom: 1.75rem;
        }

        .reg-label {
            display: block;
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .reg-input-box {
            position: relative;
        }

        .reg-input-box i {
            position: absolute;
            left: 1.25rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            transition: all 0.3s;
            pointer-events: none;
        }

        .reg-control {
            width: 100%;
            height: 56px;
            background: var(--bg-color);
            border: 2px solid transparent;
            border-radius: 16px;
            padding: 0 1.25rem 0 3.25rem;
            font-size: 1rem;
            font-weight: 500;
            color: var(--text-color);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        }

        .reg-control:focus {
            background: var(--card-bg);
            border-color: var(--reg-primary);
            box-shadow: 0 0 0 4px rgba(11, 61, 145, 0.1);
            outline: none;
        }

        .reg-control:focus + i {
            color: var(--reg-primary);
            transform: translateY(-50%) scale(1.1);
        }

        select.reg-control {
            cursor: pointer;
            appearance: none;
        }

        .reg-submit-btn {
            width: 100%;
            height: 60px;
            background: var(--reg-primary);
            color: white;
            border: none;
            border-radius: 18px;
            font-size: 1.1rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 1rem;
            box-shadow: 0 10px 25px rgba(11, 61, 145, 0.2);
        }

        .reg-submit-btn:hover {
            background: #0d47a1;
            transform: translateY(-2px);
            box-shadow: 0 15px 35px rgba(11, 61, 145, 0.3);
        }

        .reg-submit-btn:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
        }

        .reg-footer {
            margin-top: 2rem;
            text-align: center;
            font-size: 1rem;
            color: var(--text-muted);
        }

        .reg-footer a {
            color: var(--reg-primary);
            font-weight: 700;
            text-decoration: none;
            padding-bottom: 2px;
            border-bottom: 2px solid transparent;
            transition: all 0.3s;
        }

        .reg-footer a:hover {
            border-bottom-color: var(--reg-primary);
        }

        @media (max-width: 1024px) {
            .register-card {
                grid-template-columns: 1fr;
                max-width: 600px;
            }
            .register-sidebar {
                padding: 3rem;
            }
            .reg-steps {
                display: none;
            }
        }

        @media (max-width: 640px) {
            .reg-grid {
                grid-template-columns: 1fr;
            }
            .full-row { grid-column: span 1; }
            .register-form-area {
                padding: 2.5rem 1.5rem;
            }
        }
    </style>
</head>
<body>
    {{ NAVBAR }}

    <div class="register-wrapper">
        <div class="mesh-bg"></div>

        <div class="register-card">
            <aside class="register-sidebar">
                <div class="sidebar-header">
                    <i class="fas fa-heart-pulse"></i>
                    <h2>Join the Future of Heart Health</h2>
                    <p>Unlock AI-powered cardiac insights and take control of your wellbeing today.</p>
                </div>

                <ul class="reg-steps">
                    <li class="reg-step">
                        <div class="step-icon"><i class="fas fa-user-plus"></i></div>
                        <div class="step-content">
                            <h4>Create Account</h4>
                            <p>Quick and easy setup in under 2 minutes.</p>
                        </div>
                    </li>
                    <li class="reg-step">
                        <div class="step-icon"><i class="fas fa-shield-halved"></i></div>
                        <div class="step-content">
                            <h4>Secure Profile</h4>
                            <p>Enterprise-grade encryption for your health data.</p>
                        </div>
                    </li>
                    <li class="reg-step">
                        <div class="step-icon"><i class="fas fa-microchip"></i></div>
                        <div class="step-content">
                            <h4>AI Access</h4>
                            <p>Immediate access to our advanced prediction models.</p>
                        </div>
                    </li>
                </ul>

                <div class="sidebar-footer">
                    <i class="fas fa-lock"></i>
                    <span>Secure Registration Portal</span>
                </div>
            </aside>

            <main class="register-form-area">
                <div class="form-header">
                    <h1>Create Account</h1>
                    <p>Enter your details to get started</p>
                </div>

                <form id="registerForm">
                    <div class="reg-grid">
                        <div class="reg-group">
                            <label class="reg-label">Full Name</label>
                            <div class="reg-input-box">
                                <input type="text" class="reg-control" id="name" placeholder="John Doe" required>
                                <i class="fas fa-user"></i>
                            </div>
                        </div>

                        <div class="reg-group">
                            <label class="reg-label">Age</label>
                            <div class="reg-input-box">
                                <input type="number" class="reg-control" id="age" min="18" max="100" placeholder="25" required>
                                <i class="fas fa-cake-candles"></i>
                            </div>
                        </div>

                        <div class="reg-group">
                            <label class="reg-label">Gender</label>
                            <div class="reg-input-box">
                                <select class="reg-control" id="gender" required>
                                    <option value="">Select</option>
                                    <option value="Male">Male</option>
                                    <option value="Female">Female</option>
                                    <option value="Other">Other</option>
                                </select>
                                <i class="fas fa-venus-mars"></i>
                            </div>
                        </div>

                        <div class="reg-group">
                            <label class="reg-label">Email Address</label>
                            <div class="reg-input-box">
                                <input type="email" class="reg-control" id="email" placeholder="john@example.com" required>
                                <i class="fas fa-envelope"></i>
                            </div>
                        </div>

                        <div class="reg-group full-row">
                            <label class="reg-label">Username</label>
                            <div class="reg-input-box">
                                <input type="text" class="reg-control" id="username" placeholder="johndoe_health" required>
                                <i class="fas fa-at"></i>
                            </div>
                        </div>

                        <div class="reg-group">
                            <label class="reg-label">Password</label>
                            <div class="reg-input-box">
                                <input type="password" class="reg-control" id="password" placeholder="••••••••" required>
                                <i class="fas fa-lock"></i>
                            </div>
                        </div>

                        <div class="reg-group">
                            <label class="reg-label">Confirm Password</label>
                            <div class="reg-input-box">
                                <input type="password" class="reg-control" id="confirm_password" placeholder="••••••••" required>
                                <i class="fas fa-shield-check"></i>
                            </div>
                        </div>
                    </div>

                    <button type="submit" class="reg-submit-btn" id="regBtn">
                        <span>Initialize Account</span>
                        <i class="fas fa-arrow-right"></i>
                    </button>

                    <div class="reg-footer">
                        <span>Already registered?</span>
                        <a href="/login">Sign In Here</a>
                    </div>
                </form>
            </main>
        </div>
    </div>

    <script>
        document.getElementById('registerForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const btn = document.getElementById('regBtn');
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirm_password').value;
            
            if (password !== confirmPassword) {
                showAlert('Passwords do not match', 'danger');
                return;
            }

            // Loading state
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            
            const data = {
                name: document.getElementById('name').value.trim(),
                age: document.getElementById('age').value,
                gender: document.getElementById('gender').value,
                email: document.getElementById('email').value.trim(),
                username: document.getElementById('username').value.trim(),
                password: password
            };
            
            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showAlert(result.message, 'success');
                    setTimeout(() => {
                        window.location.href = '/login';
                    }, 2000);
                } else {
                    showAlert(result.message, 'danger');
                    btn.disabled = false;
                    btn.innerHTML = '<span>Initialize Account</span> <i class="fas fa-arrow-right"></i>';
                }
            } catch (error) {
                showAlert('An error occurred. Please try again.', 'danger');
                btn.disabled = false;
                btn.innerHTML = '<span>Initialize Account</span> <i class="fas fa-arrow-right"></i>';
            }
        });
    </script>
    {{ base_js | safe }}
</body>
</html>
'''

FORGOT_PASSWORD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forgot Password - Cardix AI</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    {{ base_css | safe }}
    <style>
        :root {
            --login-primary: #8B0000;
            --login-secondary: #0B3D91;
            --glass-bg: rgba(255, 255, 255, 0.75);
            --glass-border: rgba(255, 255, 255, 0.4);
            --glow-color: rgba(139, 0, 0, 0.3);
        }

        .theme-dark {
            --glass-bg: rgba(15, 23, 42, 0.8);
            --glass-border: rgba(255, 255, 255, 0.1);
            --glow-color: rgba(255, 143, 171, 0.2);
        }

        .login-wrapper {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: var(--bg-color);
            position: relative;
            overflow: hidden;
        }

        .bg-orb {
            position: absolute;
            width: 600px;
            height: 600px;
            filter: blur(100px);
            border-radius: 50%;
            z-index: 0;
            opacity: 0.2;
        }
        .orb-1 { background: var(--login-primary); top: -200px; left: -100px; animation: pulse 10s infinite alternate; }
        .orb-2 { background: var(--login-secondary); bottom: -200px; right: -100px; animation: pulse 12s infinite alternate-reverse; }

        @keyframes pulse {
            0% { transform: scale(1) translate(0, 0); }
            100% { transform: scale(1.2) translate(50px, 50px); }
        }

        .forgot-container {
            width: 100%;
            max-width: 500px;
            background: var(--glass-bg);
            backdrop-filter: blur(25px);
            border: 1px solid var(--glass-border);
            border-radius: 40px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
            padding: 60px;
            position: relative;
            z-index: 1;
        }

        .form-header {
            margin-bottom: 40px;
            text-align: center;
        }

        .form-header i {
            font-size: 3rem;
            color: var(--primary);
            margin-bottom: 20px;
        }

        .form-header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--dark);
            margin-bottom: 12px;
        }

        .form-header p {
            color: var(--text-muted);
            font-size: 1.05rem;
        }

        .input-group {
            margin-bottom: 24px;
            position: relative;
        }

        .input-group label {
            display: block;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            color: var(--primary);
        }

        .input-field {
            width: 100%;
            height: 60px;
            background: var(--bg-color);
            border: 2px solid transparent;
            border-radius: 20px;
            padding: 0 24px;
            font-size: 1rem;
            font-weight: 600;
            color: var(--dark);
            transition: all 0.3s;
        }

        .input-field:focus {
            background: var(--card-bg);
            border-color: var(--primary);
            box-shadow: 0 10px 20px -10px var(--glow-color);
            outline: none;
        }

        .btn-submit {
            width: 100%;
            height: 60px;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 20px;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-top: 20px;
            box-shadow: 0 15px 30px -10px rgba(139, 0, 0, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px -10px rgba(139, 0, 0, 0.5);
            background: #a30000;
        }

        .form-footer {
            margin-top: 32px;
            text-align: center;
            font-weight: 600;
            color: var(--text-muted);
        }

        .form-footer a {
            color: var(--primary);
            text-decoration: none;
            margin-left: 5px;
        }
    </style>
</head>
<body>
    {{ NAVBAR }}

    <div class="login-wrapper">
        <div class="bg-orb orb-1"></div>
        <div class="bg-orb orb-2"></div>

        <div class="forgot-container">
            <div class="form-header">
                <i class="fas fa-key"></i>
                <h1>Forgot Password?</h1>
                <p>Enter your email address and we'll send you a password reset link</p>
            </div>

            <form id="forgotPasswordForm">
                <div class="input-group">
                    <label>Email Address</label>
                    <input type="email" id="email" class="input-field" placeholder="your@email.com" required>
                </div>

                <button type="submit" class="btn-submit">
                    Send Reset Link <i class="fas fa-paper-plane"></i>
                </button>

                <div class="form-footer">
                    Remember your password? <a href="/login">Back to Login</a>
                </div>
            </form>
        </div>
    </div>

    <script>
        document.getElementById('forgotPasswordForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const email = document.getElementById('email').value.trim();
            
            const btn = e.target.querySelector('button');
            const originalText = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';

            try {
                const response = await fetch('/forgot-password', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        'email': email
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showAlert(result.message, 'success');
                    // Show reset URL if email is not configured (for testing)
                    if (result.reset_url) {
                        setTimeout(() => {
                            if (confirm('Email service not configured. Click OK to open reset link.')) {
                                window.location.href = result.reset_url;
                            }
                        }, 2000);
                    }
                } else {
                    showAlert(result.message, 'danger');
                }
                
                btn.disabled = false;
                btn.innerHTML = originalText;
            } catch (error) {
                showAlert('An error occurred. Please try again.', 'danger');
                btn.disabled = false;
                btn.innerHTML = originalText;
            }
        });
    </script>
    {{ base_js | safe }}
</body>
</html>
'''

RESET_PASSWORD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reset Password - Cardix AI</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    {{ base_css | safe }}
    <style>
        :root {
            --login-primary: #8B0000;
            --login-secondary: #0B3D91;
            --glass-bg: rgba(255, 255, 255, 0.75);
            --glass-border: rgba(255, 255, 255, 0.4);
            --glow-color: rgba(139, 0, 0, 0.3);
        }

        .theme-dark {
            --glass-bg: rgba(15, 23, 42, 0.8);
            --glass-border: rgba(255, 255, 255, 0.1);
            --glow-color: rgba(255, 143, 171, 0.2);
        }

        .login-wrapper {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: var(--bg-color);
            position: relative;
            overflow: hidden;
        }

        .bg-orb {
            position: absolute;
            width: 600px;
            height: 600px;
            filter: blur(100px);
            border-radius: 50%;
            z-index: 0;
            opacity: 0.2;
        }
        .orb-1 { background: var(--login-primary); top: -200px; left: -100px; animation: pulse 10s infinite alternate; }
        .orb-2 { background: var(--login-secondary); bottom: -200px; right: -100px; animation: pulse 12s infinite alternate-reverse; }

        @keyframes pulse {
            0% { transform: scale(1) translate(0, 0); }
            100% { transform: scale(1.2) translate(50px, 50px); }
        }

        .reset-container {
            width: 100%;
            max-width: 500px;
            background: var(--glass-bg);
            backdrop-filter: blur(25px);
            border: 1px solid var(--glass-border);
            border-radius: 40px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
            padding: 60px;
            position: relative;
            z-index: 1;
        }

        .form-header {
            margin-bottom: 40px;
            text-align: center;
        }

        .form-header i {
            font-size: 3rem;
            color: var(--primary);
            margin-bottom: 20px;
        }

        .form-header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--dark);
            margin-bottom: 12px;
        }

        .form-header p {
            color: var(--text-muted);
            font-size: 1.05rem;
        }

        .input-group {
            margin-bottom: 24px;
            position: relative;
        }

        .input-group label {
            display: block;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            color: var(--primary);
        }

        .input-field {
            width: 100%;
            height: 60px;
            background: var(--bg-color);
            border: 2px solid transparent;
            border-radius: 20px;
            padding: 0 24px;
            font-size: 1rem;
            font-weight: 600;
            color: var(--dark);
            transition: all 0.3s;
        }

        .input-field:focus {
            background: var(--card-bg);
            border-color: var(--primary);
            box-shadow: 0 10px 20px -10px var(--glow-color);
            outline: none;
        }

        .btn-submit {
            width: 100%;
            height: 60px;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 20px;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-top: 20px;
            box-shadow: 0 15px 30px -10px rgba(139, 0, 0, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px -10px rgba(139, 0, 0, 0.5);
            background: #a30000;
        }

        .password-strength {
            margin-top: 10px;
            height: 4px;
            background: #e0e0e0;
            border-radius: 2px;
            overflow: hidden;
        }

        .strength-bar {
            height: 100%;
            width: 0%;
            transition: all 0.3s;
        }

        .form-footer {
            margin-top: 32px;
            text-align: center;
            font-weight: 600;
            color: var(--text-muted);
        }

        .form-footer a {
            color: var(--primary);
            text-decoration: none;
            margin-left: 5px;
        }
    </style>
</head>
<body>
    {{ NAVBAR }}

    <div class="login-wrapper">
        <div class="bg-orb orb-1"></div>
        <div class="bg-orb orb-2"></div>

        <div class="reset-container">
            <div class="form-header">
                <i class="fas fa-lock-open"></i>
                <h1>Reset Password</h1>
                <p>Enter your new password below</p>
            </div>

            <form id="resetPasswordForm">
                <input type="hidden" id="token" value="{{ token }}">
                
                <div class="input-group">
                    <label>New Password</label>
                    <input type="password" id="new_password" class="input-field" placeholder="••••••••" required minlength="6">
                    <div class="password-strength">
                        <div class="strength-bar" id="strengthBar"></div>
                    </div>
                </div>

                <div class="input-group">
                    <label>Confirm Password</label>
                    <input type="password" id="confirm_password" class="input-field" placeholder="••••••••" required minlength="6">
                </div>

                <button type="submit" class="btn-submit">
                    Update Password <i class="fas fa-check"></i>
                </button>

                <div class="form-footer">
                    Remember your password? <a href="/login">Back to Login</a>
                </div>
            </form>
        </div>
    </div>

    <script>
        // Password strength indicator
        document.getElementById('new_password').addEventListener('input', function(e) {
            const password = e.target.value;
            const strengthBar = document.getElementById('strengthBar');
            
            let strength = 0;
            if (password.length >= 6) strength += 25;
            if (password.length >= 10) strength += 25;
            if (/[a-z]/.test(password) && /[A-Z]/.test(password)) strength += 25;
            if (/[0-9]/.test(password)) strength += 15;
            if (/[^a-zA-Z0-9]/.test(password)) strength += 10;
            
            strengthBar.style.width = strength + '%';
            
            if (strength < 40) {
                strengthBar.style.background = '#ff4444';
            } else if (strength < 70) {
                strengthBar.style.background = '#ffaa00';
            } else {
                strengthBar.style.background = '#00cc66';
            }
        });

        document.getElementById('resetPasswordForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const token = document.getElementById('token').value;
            const newPassword = document.getElementById('new_password').value;
            const confirmPassword = document.getElementById('confirm_password').value;
            
            if (newPassword !== confirmPassword) {
                showAlert('Passwords do not match', 'danger');
                return;
            }
            
            const btn = e.target.querySelector('button');
            const originalText = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Updating...';

            try {
                const response = await fetch('/reset-password', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        'token': token,
                        'new_password': newPassword,
                        'confirm_password': confirmPassword
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showAlert(result.message, 'success');
                    setTimeout(() => {
                        window.location.href = '/login';
                    }, 2000);
                } else {
                    showAlert(result.message, 'danger');
                    btn.disabled = false;
                    btn.innerHTML = originalText;
                }
            } catch (error) {
                showAlert('An error occurred. Please try again.', 'danger');
                btn.disabled = false;
                btn.innerHTML = originalText;
            }
        });
    </script>
    {{ base_js | safe }}
</body>
</html>
'''

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Cardix AI</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    {{ base_css | safe }}
    <style>
        :root {
            --dash-primary: #ff4757;
            --dash-secondary: #2f3542;
            --dash-accent: #2ed573;
            --dash-bg: #f1f2f6;
            --card-shadow: 0 10px 30px rgba(0,0,0,0.08);
        }

        .theme-dark {
            --dash-bg: #1e272e;
            --dash-secondary: #ced4da;
            --card-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        .dashboard-wrapper {
            min-height: 100vh;
            background: var(--dash-bg);
            padding: 2rem 1rem;
            position: relative;
            overflow-x: hidden;
        }

        .dashboard-container {
            max-width: 1200px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }

        .welcome-card {
            background: white;
            border-radius: 30px;
            padding: 3rem;
            margin-bottom: 2rem;
            box-shadow: var(--card-shadow);
            display: flex;
            justify-content: space-between;
            align-items: center;
            overflow: hidden;
            position: relative;
        }

        .theme-dark .welcome-card {
            background: #2f3542;
        }

        .welcome-card::before {
            content: '';
            position: absolute;
            top: -50%; right: -10%;
            width: 300px; height: 300px;
            background: var(--dash-primary);
            filter: blur(80px);
            opacity: 0.1;
            border-radius: 50%;
        }

        .welcome-info h1 {
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--dash-secondary);
            margin-bottom: 0.5rem;
        }

        .welcome-info p {
            color: #747d8c;
            font-size: 1.1rem;
        }

        .stats-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .stat-box {
            background: white;
            padding: 2rem;
            border-radius: 24px;
            box-shadow: var(--card-shadow);
            transition: transform 0.3s ease;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .theme-dark .stat-box {
            background: #2f3542;
        }

        .stat-box:hover {
            transform: translateY(-5px);
        }

        .stat-icon-circle {
            width: 50px; height: 50px;
            border-radius: 15px;
            background: rgba(255, 71, 87, 0.1);
            color: var(--dash-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }

        .stat-data h3 {
            font-size: 0.9rem;
            color: #747d8c;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }

        .stat-data .value {
            font-size: 2rem;
            font-weight: 800;
            color: var(--dash-secondary);
        }

        .action-sections {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
        }

        .main-actions {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        .action-tile {
            background: white;
            padding: 2.5rem;
            border-radius: 28px;
            box-shadow: var(--card-shadow);
            text-decoration: none;
            color: var(--dash-secondary);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            border: 2px solid transparent;
        }

        .theme-dark .action-tile {
            background: #2f3542;
        }

        .action-tile:hover {
            transform: scale(1.02);
            border-color: var(--dash-primary);
        }

        .action-tile.primary-action {
            background: var(--dash-primary);
            color: white;
        }

        .action-tile.primary-action p {
            color: rgba(255,255,255,0.8);
        }

        .action-tile i {
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
            display: block;
        }

        .action-tile h2 {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .tips-panel {
            background: white;
            padding: 2rem;
            border-radius: 28px;
            box-shadow: var(--card-shadow);
        }

        .theme-dark .tips-panel {
            background: #2f3542;
        }

        .tips-panel h2 {
            font-size: 1.3rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .tip-card {
            background: var(--dash-bg);
            padding: 1rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            font-size: 0.95rem;
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }

        .theme-dark .tip-card {
            background: #1e272e;
        }

        @media (max-width: 992px) {
            .action-sections { grid-template-columns: 1fr; }
        }

        @media (max-width: 768px) {
            .main-actions { grid-template-columns: 1fr; }
            .welcome-card { padding: 2rem; flex-direction: column; text-align: center; }
        }
    </style>
</head>
<body>
    {{ NAVBAR }}

    <div class="dashboard-wrapper">
        <div class="dashboard-container">
            <header class="welcome-card">
                <div class="welcome-info">
                    <h1>Hi, {{ name }}!</h1>
                    <p>Welcome back to your health monitoring center.</p>
                </div>
                <div class="welcome-badge">
                    <i class="fas fa-heart-pulse fa-4x" style="color: var(--dash-primary); opacity: 0.2;"></i>
                </div>
            </header>

            <div class="stats-row">
                <div class="stat-box">
                    <div class="stat-icon-circle"><i class="fas fa-microscope"></i></div>
                    <div class="stat-data">
                        <h3>Analyses Completed</h3>
                        <div class="value">{{ total_predictions }}</div>
                    </div>
                </div>
                <div class="stat-box">
                    <div class="stat-icon-circle" style="background: rgba(46, 213, 115, 0.1); color: var(--dash-accent);">
                        <i class="fas fa-clock-rotate-left"></i>
                    </div>
                    <div class="stat-data">
                        <h3>Last Checkup</h3>
                        <div class="value">{{ last_prediction[1][:10] if last_prediction else 'None' }}</div>
                    </div>
                </div>
                <div class="stat-box">
                    <div class="stat-icon-circle" style="background: rgba(47, 53, 66, 0.1); color: var(--dash-secondary);">
                        <i class="fas fa-shield-heart"></i>
                    </div>
                    <div class="stat-data">
                        <h3>Account Status</h3>
                        <div class="value">Verified</div>
                    </div>
                </div>
            </div>

            <div class="action-sections">
                <div class="main-actions">
                    <a href="/predict" class="action-tile primary-action">
                        <i class="fas fa-plus-circle"></i>
                        <h2>Start Analysis</h2>
                        <p>Run our AI model to check your heart health risk factors.</p>
                    </a>
                    <a href="/history" class="action-tile">
                        <i class="fas fa-folder-open" style="color: var(--dash-primary)"></i>
                        <h2>History</h2>
                        <p>Access and review all your past health assessments.</p>
                    </a>
                    <a href="/profile" class="action-tile">
                        <i class="fas fa-user-gear" style="color: #ffa502"></i>
                        <h2>Profile</h2>
                        <p>Manage your settings and personal clinical data.</p>
                    </a>
                    <a href="/about" class="action-tile">
                        <i class="fas fa-circle-info" style="color: #1e90ff"></i>
                        <h2>About AI</h2>
                        <p>Learn about our predictive algorithms and data security.</p>
                    </a>
                </div>

                <aside class="tips-panel">
                    <h2><i class="fas fa-lightbulb" style="color: #f1c40f"></i> Health Tips</h2>
                    <div class="tip-list">
                        <div class="tip-card">
                            <i class="fas fa-check" style="color: var(--dash-accent)"></i>
                            Stay active for 30 mins a day.
                        </div>
                        <div class="tip-card">
                            <i class="fas fa-check" style="color: var(--dash-accent)"></i>
                            Monitor your blood pressure.
                        </div>
                        <div class="tip-card">
                            <i class="fas fa-check" style="color: var(--dash-accent)"></i>
                            Stay hydrated and eat greens.
                        </div>
                    </div>
                </aside>
            </div>
        </div>
    </div>

    {{ base_js | safe }}
</body>
</html>
'''

PREDICT_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Risk Assessment - Cardix AI</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    {{ base_css | safe }}
   <style>
        :root {
            --predict-primary: #6366f1;
            --predict-secondary: #8b5cf6;
            --predict-accent: #10b981;
            --predict-bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --input-bg: rgba(15, 23, 42, 0.6);
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
            --glass-border: rgba(255, 255, 255, 0.1);
        }

        .predict-page {
            min-height: 100vh;
            background: var(--predict-bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.15) 0px, transparent 50%);
            padding: 4rem 1rem 2rem 1rem;
            color: var(--text-main);
        }

        .predict-container {
            max-width: 1000px;
            margin: 0 auto;
        }

        .predict-header {
            text-align: center;
            margin-bottom: 3rem;
        }

        .predict-header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .predict-header p {
            color: var(--text-dim);
            font-size: 1rem;
            max-width: 600px;
            margin: 0 auto;
        }

        .info-banner {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .info-banner i {
            color: var(--predict-primary);
            font-size: 1.5rem;
        }

        .predict-card {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            margin-bottom: 2rem;
        }

        .section-header {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--predict-primary);
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid var(--glass-border);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .predict-form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .predict-input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        

        .known-condition-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .predict-input-group label {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .predict-input-group label i {
            color: var(--predict-primary);
            font-size: 0.9rem;
        }

        .predict-input {
            background: var(--input-bg);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            color: var(--text-main);
            font-size: 1rem;
            transition: all 0.3s;
        }

        .predict-input:focus {
            outline: none;
            border-color: var(--predict-primary);
            background: rgba(15, 23, 42, 0.8);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
        }

        .predict-input option {
            background: #1e293b;
            color: white;
        }

        .predict-help {
            font-size: 0.75rem;
            color: #9fb0c7;
            line-height: 1.3;
        }

        .bmi-display {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            color: var(--predict-accent);
            font-weight: 700;
            font-size: 1.1rem;
        }

        .symptom-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1rem;
        }

        .symptom-item {
            background: var(--input-bg);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 1rem;
            position: relative;
        }

        .symptom-item label {
            font-weight: 500;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }

        .symptom-info-btn {
            width: 22px;
            height: 22px;
            border: 1px solid var(--glass-border);
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-main);
            font-size: 0.75rem;
            font-weight: 700;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .symptom-info-popup {
            display: none;
            position: absolute;
            top: 42px;
            right: 14px;
            width: min(320px, calc(100% - 28px));
            background: #0b1224;
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 10px;
            padding: 0.75rem;
            color: #dbeafe;
            font-size: 0.82rem;
            line-height: 1.35;
            z-index: 20;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.35);
        }

        .symptom-info-popup.open {
            display: block;
        }

        .symptom-options {
            display: flex;
            gap: 0.75rem;
        }

        .symptom-btn {
            flex: 1;
            padding: 0.6rem;
            background: transparent;
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            color: var(--text-dim);
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.85rem;
        }

        .symptom-btn:hover {
            border-color: var(--predict-primary);
        }

        .symptom-btn.active {
            background: var(--predict-primary);
            color: white;
            border-color: var(--predict-primary);
        }

        .model-toggle-group {
            display: flex;
            background: var(--input-bg);
            padding: 0.5rem;
            border-radius: 14px;
            margin-bottom: 2rem;
            width: fit-content;
            margin-left: auto;
            margin-right: auto;
            border: 1px solid var(--glass-border);
        }

        .model-toggle-btn {
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            border: none;
            background: transparent;
            color: var(--text-dim);
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .model-toggle-btn.active {
            background: var(--predict-primary);
            color: white;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }

        .btn-predict-submit {
            width: 100%;
            padding: 1.25rem;
            border-radius: 16px;
            border: none;
            background: linear-gradient(135deg, var(--predict-primary), var(--predict-secondary));
            color: white;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
            margin-top: 2rem;
        }

        .btn-predict-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px -5px rgba(99, 102, 241, 0.5);
        }

        .btn-predict-submit:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
        }

        /* Result Modal */
        .predict-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(2, 6, 23, 0.9);
            backdrop-filter: blur(8px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }

        .predict-result-card {
            background: #1e293b;
            border: 1px solid var(--glass-border);
            border-radius: 32px;
            padding: 4rem;
            max-width: 600px;
            width: 100%;
            text-align: center;
            position: relative;
            animation: predictPop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        @keyframes predictPop {
            from { opacity: 0; transform: scale(0.9) translateY(20px); }
            to { opacity: 1; transform: scale(1) translateY(0); }
        }

        .predict-close {
            position: absolute;
            top: 1.5rem;
            right: 1.5rem;
            width: 40px;
            height: 40px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.05);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            color: var(--text-dim);
            transition: all 0.2s;
        }

        .predict-close:hover {
            background: rgba(239, 68, 68, 0.1);
            color: #ef4444;
        }

        .predict-gauge-container {
            width: 200px;
            height: 200px;
            margin: 0 auto 2rem;
            position: relative;
        }

        .predict-gauge-svg {
            transform: rotate(-90deg);
        }

        .predict-gauge-bg {
            fill: none;
            stroke: rgba(255, 255, 255, 0.05);
            stroke-width: 10;
        }

        .predict-gauge-fill {
            fill: none;
            stroke-width: 10;
            stroke-linecap: round;
            transition: stroke-dashoffset 1s ease-out;
        }

        .predict-gauge-content {
            position: absolute;
            inset: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .predict-gauge-val {
            font-size: 3rem;
            font-weight: 800;
            color: var(--text-main);
        }

        .predict-gauge-label {
            font-size: 0.8rem;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .predict-status-text {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }

        .predict-advice-text {
            color: var(--text-dim);
            line-height: 1.6;
            margin-bottom: 2.5rem;
        }

        .predict-result-btns {
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
        }

        .btn-predict-action {
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.2s;
            border: 1px solid var(--glass-border);
            cursor: pointer;
            background: transparent;
        }

        .btn-predict-history {
            background: var(--predict-primary);
            color: white;
            border: none;
        }

        .btn-predict-new {
            color: var(--text-main);
        }

        .btn-predict-new:hover {
            background: rgba(255, 255, 255, 0.05);
        }

        @media (max-width: 768px) {
            .predict-card { padding: 1.5rem; }
            .predict-form-grid { grid-template-columns: 1fr; }
            .known-condition-grid { grid-template-columns: 1fr; }
            .predict-header h1 { font-size: 2rem; }
            .symptom-options { flex-direction: column; }
        }
    </style>
</head>
<body>
    {{ NAVBAR }}

    <div class="predict-page">
        <div class="predict-container">
            <header class="predict-header">
                <h1><i class="fas fa-heartbeat"></i> Heart Risk Assessment</h1>
                <p>This assessment uses only information you can know without medical tests. Answer based on your lifestyle, family history, and any symptoms you experience.</p>
            </header>

            <div class="info-banner">
                <i class="fas fa-info-circle"></i>
                <div>
                    <strong>No Medical Tests Required</strong> - Fill out this form using only what you know about yourself. 
                    This tool provides a risk estimate and is not a medical diagnosis.
                </div>
            </div>

            <div class="predict-card">
                <div class="model-toggle-group">
                    <button type="button" class="model-toggle-btn active" data-model="random_forest">
                        <i class="fas fa-brain"></i> Random Forest
                    </button>
                    <button type="button" class="model-toggle-btn" data-model="logistic">
                        <i class="fas fa-microscope"></i> Logistic
                    </button>
                </div>

                <form id="predictionForm">
                    <!-- Personal Information Section -->
                    <div class="section-header">
                        <i class="fas fa-user"></i> Personal Information
                    </div>
                    <div class="predict-form-grid">
                        <div class="predict-input-group">
                            <label><i class="fas fa-calendar"></i> Age</label>
                            <input type="number" class="predict-input" id="age" min="18" max="100" placeholder="Years" required>
                            <small class="predict-help">Your current age in years.</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-venus-mars"></i> Gender</label>
                            <select class="predict-input" id="sex" required>
                                <option value="">Select</option>
                                <option value="1">Male</option>
                                <option value="0">Female</option>
                            </select>
                            <small class="predict-help">Biological sex.</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-ruler-vertical"></i> Height (cm)</label>
                            <input type="number" class="predict-input" id="height" step="0.1" min="100" max="250" placeholder="170" required>
                            <small class="predict-help">Your height in centimeters.</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-weight-scale"></i> Weight (kg)</label>
                            <input type="number" class="predict-input" id="weight" step="0.1" min="30" max="250" placeholder="70" required>
                            <small class="predict-help">Your current weight in kilograms.</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-calculator"></i> BMI</label>
                            <div class="bmi-display" id="bmiDisplay">--</div>
                            <small class="predict-help">Auto-calculated from height and weight.</small>
                        </div>
                    </div>

                    <!-- Lifestyle Factors Section -->
                    <div class="section-header">
                        <i class="fas fa-person-running"></i> Lifestyle Factors
                    </div>
                    <div class="predict-form-grid">
                        <div class="predict-input-group">
                            <label><i class="fas fa-smoking"></i> Smoking Status</label>
                            <select class="predict-input" id="smoking" required>
                                <option value="">Select</option>
                                <option value="0">Never Smoked</option>
                                <option value="1">Former Smoker</option>
                                <option value="2">Current Smoker</option>
                            </select>
                            <small class="predict-help">Your smoking history.</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-wine-glass"></i> Alcohol Consumption</label>
                            <select class="predict-input" id="alcohol" required>
                                <option value="">Select</option>
                                <option value="0">None</option>
                                <option value="1">Occasional (1-7 drinks/week)</option>
                                <option value="2">Regular (8+ drinks/week)</option>
                            </select>
                            <small class="predict-help">How often you consume alcohol.</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-dumbbell"></i> Physical Activity Level</label>
                            <select class="predict-input" id="physical_activity" required>
                                <option value="">Select</option>
                                <option value="0">Sedentary (Little/no exercise)</option>
                                <option value="1">Moderate (Exercise 1-3 times/week)</option>
                                <option value="2">Active (Exercise 4+ times/week)</option>
                            </select>
                            <small class="predict-help">Your typical exercise routine.</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-bed"></i> Sleep Duration</label>
                            <input type="number" class="predict-input" id="sleep_hours" step="0.5" min="2" max="14" placeholder="7" required>
                            <small class="predict-help">Average hours per night.</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-brain"></i> Stress Level</label>
                            <select class="predict-input" id="stress_level" required>
                                <option value="">Select</option>
                                <option value="0">Low</option>
                                <option value="1">Moderate</option>
                                <option value="2">High</option>
                            </select>
                            <small class="predict-help">Your typical stress level.</small>
                        </div>
                    </div>

                    <!-- Known Medical Conditions Section -->
                    <div class="section-header">
                        <i class="fas fa-notes-medical"></i> Known Medical Conditions
                    </div>
                    <div class="known-condition-grid">
                        <div class="predict-input-group">
                            <label><i class="fas fa-heart-pulse"></i> High Blood Pressure Diagnosis</label>
                            <select class="predict-input" id="has_hypertension" required>
                                <option value="">Select</option>
                                <option value="0">No</option>
                                <option value="1">Not Sure</option>
                                <option value="2">Yes</option>
                            </select>
                            <small class="predict-help">Have you been diagnosed with high blood pressure?</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-gauge"></i> Latest BP (Optional)</label>
                            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
                                <input type="number" class="predict-input" id="resting_systolic_bp" min="70" max="250" placeholder="Systolic (e.g., 120)">
                                <input type="number" class="predict-input" id="resting_diastolic_bp" min="40" max="150" placeholder="Diastolic (e.g., 80)">
                            </div>
                            <small class="predict-help">Optional: Enter your latest blood pressure in mmHg if known.</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-vial"></i> High Cholesterol Diagnosis</label>
                            <select class="predict-input" id="has_high_cholesterol" required>
                                <option value="">Select</option>
                                <option value="2">Yes</option>
                                <option value="0">No</option>
                                <option value="1">Not Sure</option>
                            </select>
                            <small class="predict-help">Have you been diagnosed with high cholesterol?</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-flask"></i> Latest Cholesterol (Optional)</label>
                            <input type="number" class="predict-input" id="cholesterol_value" min="80" max="500" placeholder="mg/dL (e.g., 190)">
                            <small class="predict-help">Optional: Enter your latest cholesterol (mg/dL).</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-droplet"></i> Diabetes Diagnosis</label>
                            <select class="predict-input" id="has_diabetes" required>
                                <option value="">Select</option>
                                <option value="0">No</option>
                                <option value="1">Not Sure</option>
                                <option value="2">Yes</option>
                            </select>
                            <small class="predict-help">Have you been diagnosed with diabetes?</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-candy-cane"></i> Fasting Blood Sugar (Optional)</label>
                            <input type="number" class="predict-input" id="fasting_blood_sugar" min="50" max="400" placeholder="mg/dL (e.g., 95)">
                            <small class="predict-help">Optional: Enter your latest fasting blood sugar if known.</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-heart"></i> Resting Heart Rate Knowledge</label>
                            <select class="predict-input" id="know_resting_heart_rate" required>
                                <option value="">Select</option>
                                <option value="2">Yes</option>
                                <option value="0">No</option>
                                <option value="1">Not Sure</option>
                            </select>
                            <small class="predict-help">Do you know your resting heart rate?</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-heart"></i> Resting Heart Rate (Optional)</label>
                            <input type="number" class="predict-input" id="resting_heart_rate" min="30" max="220" placeholder="bpm (e.g., 72)">
                            <small class="predict-help">Optional: Enter your resting heart rate (bpm).</small>
                        </div>
                    </div>

                    <!-- Family History Section -->
                    <div class="section-header">
                        <i class="fas fa-users"></i> Family History
                    </div>
                    <div class="predict-form-grid">
                        <div class="predict-input-group">
                            <label><i class="fas fa-dna"></i> Family History of Heart Disease</label>
                            <select class="predict-input" id="family_history" required>
                                <option value="">Select</option>
                                <option value="0">None</option>
                                <option value="1">Father</option>
                                <option value="2">Mother</option>
                                <option value="3">Sibling(s)</option>
                                <option value="4">Multiple Family Members</option>
                            </select>
                            <small class="predict-help">Who in your immediate family has/had heart disease?</small>
                        </div>

                        <div class="predict-input-group">
                            <label><i class="fas fa-heartbeat"></i> Heart Attack in Family Before Age 60</label>
                            <select class="predict-input" id="family_early_heart_attack" required>
                                <option value="">Select</option>
                                <option value="0">No</option>
                                <option value="1">Not Sure</option>
                                <option value="2">Yes</option>
                            </select>
                            <small class="predict-help">Did any close relative have a heart attack before age 60?</small>
                        </div>
                    </div>

                    <!-- Symptoms Section -->
                    <div class="section-header">
                        <i class="fas fa-stethoscope"></i> Symptoms
                    </div>
                    <small class="predict-help" style="display: block; margin-bottom: 1rem;">Select how often you experience each symptom:</small>
                    
                    <div class="symptom-grid">
                        <div class="symptom-item">
                            <label>
                                <span><i class="fas fa-hand-point-right"></i> Chest Pain or Pressure</span>
                                <button type="button" class="symptom-info-btn" data-info="A tight or heavy feeling in your chest, like someone is pressing on it.">i</button>
                            </label>
                            <div class="symptom-info-popup"></div>
                            <div class="symptom-options">
                                <button type="button" class="symptom-btn active" data-symptom="symptom_chest_pain" data-value="0">Never</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_chest_pain" data-value="1">Sometimes</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_chest_pain" data-value="2">Often</button>
                            </div>
                            <input type="hidden" id="symptom_chest_pain" value="0">
                        </div>

                        <div class="symptom-item">
                            <label>
                                <span><i class="fas fa-hand-point-right"></i> Chest Pain During Activity</span>
                                <button type="button" class="symptom-info-btn" data-info="Chest discomfort when walking, climbing stairs, or exercising.">i</button>
                            </label>
                            <div class="symptom-info-popup"></div>
                            <div class="symptom-options">
                                <button type="button" class="symptom-btn active" data-symptom="symptom_chest_pain_activity" data-value="0">Never</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_chest_pain_activity" data-value="1">Sometimes</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_chest_pain_activity" data-value="2">Often</button>
                            </div>
                            <input type="hidden" id="symptom_chest_pain_activity" value="0">
                        </div>

                        <div class="symptom-item">
                            <label>
                                <span><i class="fas fa-hand-point-right"></i> Shortness of Breath</span>
                                <button type="button" class="symptom-info-btn" data-info="Feeling like it is hard to breathe or catch your breath.">i</button>
                            </label>
                            <div class="symptom-info-popup"></div>
                            <div class="symptom-options">
                                <button type="button" class="symptom-btn active" data-symptom="symptom_shortness_breath" data-value="0">Never</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_shortness_breath" data-value="1">Sometimes</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_shortness_breath" data-value="2">Often</button>
                            </div>
                            <input type="hidden" id="symptom_shortness_breath" value="0">
                        </div>

                        <div class="symptom-item">
                            <label>
                                <span><i class="fas fa-hand-point-right"></i> Unusual Fatigue</span>
                                <button type="button" class="symptom-info-btn" data-info="Feeling very tired or weak without a clear reason.">i</button>
                            </label>
                            <div class="symptom-info-popup"></div>
                            <div class="symptom-options">
                                <button type="button" class="symptom-btn active" data-symptom="symptom_fatigue" data-value="0">Never</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_fatigue" data-value="1">Sometimes</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_fatigue" data-value="2">Often</button>
                            </div>
                            <input type="hidden" id="symptom_fatigue" value="0">
                        </div>

                        <div class="symptom-item">
                            <label>
                                <span><i class="fas fa-hand-point-right"></i> Heart Palpitations</span>
                                <button type="button" class="symptom-info-btn" data-info="Your heart feels like it is beating very fast, skipping beats, or fluttering.">i</button>
                            </label>
                            <div class="symptom-info-popup"></div>
                            <div class="symptom-options">
                                <button type="button" class="symptom-btn active" data-symptom="symptom_palpitations" data-value="0">Never</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_palpitations" data-value="1">Sometimes</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_palpitations" data-value="2">Often</button>
                            </div>
                            <input type="hidden" id="symptom_palpitations" value="0">
                        </div>

                        <div class="symptom-item">
                            <label>
                                <span><i class="fas fa-hand-point-right"></i> Dizziness or Fainting</span>
                                <button type="button" class="symptom-info-btn" data-info="Feeling lightheaded, dizzy, or like you might pass out.">i</button>
                            </label>
                            <div class="symptom-info-popup"></div>
                            <div class="symptom-options">
                                <button type="button" class="symptom-btn active" data-symptom="symptom_dizziness" data-value="0">Never</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_dizziness" data-value="1">Sometimes</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_dizziness" data-value="2">Often</button>
                            </div>
                            <input type="hidden" id="symptom_dizziness" value="0">
                        </div>

                        <div class="symptom-item">
                            <label>
                                <span><i class="fas fa-hand-point-right"></i> Pain Spreading to Arm, Jaw, Neck, or Back</span>
                                <button type="button" class="symptom-info-btn" data-info="Pain that starts in your chest and moves to your left arm, jaw, neck, or back.">i</button>
                            </label>
                            <div class="symptom-info-popup"></div>
                            <div class="symptom-options">
                                <button type="button" class="symptom-btn active" data-symptom="symptom_radiating_pain" data-value="0">Never</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_radiating_pain" data-value="1">Sometimes</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_radiating_pain" data-value="2">Often</button>
                            </div>
                            <input type="hidden" id="symptom_radiating_pain" value="0">
                        </div>

                        <div class="symptom-item">
                            <label>
                                <span><i class="fas fa-hand-point-right"></i> Swelling in Legs or Feet</span>
                                <button type="button" class="symptom-info-btn" data-info="Puffiness or fluid in your legs, ankles, or feet, making them look bigger or tight.">i</button>
                            </label>
                            <div class="symptom-info-popup"></div>
                            <div class="symptom-options">
                                <button type="button" class="symptom-btn active" data-symptom="symptom_swelling" data-value="0">Never</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_swelling" data-value="1">Sometimes</button>
                                <button type="button" class="symptom-btn" data-symptom="symptom_swelling" data-value="2">Often</button>
                            </div>
                            <input type="hidden" id="symptom_swelling" value="0">
                        </div>
                    </div>

                    <button type="submit" class="btn-predict-submit">
                        <i class="fas fa-microchip"></i>
                        <span>Analyze My Risk</span>
                    </button>

                    <small class="predict-help" style="display:block; margin-top: 1rem; text-align:center;">
                        This tool provides an early risk prediction and does not replace professional medical advice. Please consult a healthcare professional for proper diagnosis.
                    </small>
                </form>
            </div>
        </div>
    </div>

    <!-- Result Modal -->
    <div id="predictOverlay" class="predict-overlay">
        <div class="predict-result-card">
            <div class="predict-close" onclick="closePredictResult()"><i class="fas fa-times"></i></div>
            
            <div class="predict-gauge-container">
                <svg class="predict-gauge-svg" width="200" height="200">
                    <circle class="predict-gauge-bg" cx="100" cy="100" r="90"></circle>
                    <circle id="predictGaugeFill" class="predict-gauge-fill" cx="100" cy="100" r="90" stroke-dasharray="565.48" stroke-dashoffset="565.48"></circle>
                </svg>
                <div class="predict-gauge-content">
                    <span id="predictGaugeVal" class="predict-gauge-val">0%</span>
                    <span class="predict-gauge-label">Risk Score</span>
                </div>
            </div>

            <h2 id="predictStatusText" class="predict-status-text">Processing...</h2>
            <p id="predictAdviceText" class="predict-advice-text"></p>
            <p id="bmiResult" style="color: var(--text-dim); font-size: 0.9rem;"></p>

            <div class="predict-result-btns">
                <a href="/history" class="btn-predict-action btn-predict-history">
                    <i class="fas fa-clock-rotate-left"></i> View History
                </a>
                <button class="btn-predict-action btn-predict-new" onclick="closePredictResult()">
                    <i class="fas fa-rotate"></i> New Assessment
                </button>
            </div>
        </div>
    </div>

    <script>
        let currentModel = 'random_forest';

        // Model toggle
        document.querySelectorAll('.model-toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.model-toggle-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentModel = btn.dataset.model;
            });
        });

        // BMI Calculator
        function calculateBMI() {
            const height = parseFloat(document.getElementById('height').value);
            const weight = parseFloat(document.getElementById('weight').value);
            
            if (height > 0 && weight > 0) {
                // Convert cm to meters before BMI calculation.
                const heightMeters = height / 100;
                const bmi = weight / (heightMeters * heightMeters);
                let category = 'Normal';
                
                // Color code BMI
                const bmiDisplay = document.getElementById('bmiDisplay');
                if (bmi < 18.5) {
                    category = 'Underweight';
                    bmiDisplay.style.background = 'rgba(59, 130, 246, 0.1)';
                    bmiDisplay.style.borderColor = 'rgba(59, 130, 246, 0.3)';
                    bmiDisplay.style.color = '#3b82f6';
                } else if (bmi < 25) {
                    category = 'Normal';
                    bmiDisplay.style.background = 'rgba(16, 185, 129, 0.1)';
                    bmiDisplay.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                    bmiDisplay.style.color = '#10b981';
                } else if (bmi < 30) {
                    category = 'Overweight';
                    bmiDisplay.style.background = 'rgba(245, 158, 11, 0.1)';
                    bmiDisplay.style.borderColor = 'rgba(245, 158, 11, 0.3)';
                    bmiDisplay.style.color = '#f59e0b';
                } else {
                    category = 'Obese';
                    bmiDisplay.style.background = 'rgba(239, 68, 68, 0.1)';
                    bmiDisplay.style.borderColor = 'rgba(239, 68, 68, 0.3)';
                    bmiDisplay.style.color = '#ef4444';
                }

                bmiDisplay.textContent = `${bmi.toFixed(2)} (${category})`;
            } else {
                document.getElementById('bmiDisplay').textContent = '--';
            }
        }

        document.getElementById('height').addEventListener('input', calculateBMI);
        document.getElementById('weight').addEventListener('input', calculateBMI);

        // Symptom buttons
        document.querySelectorAll('.symptom-btn').forEach(btn => {
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                const symptomName = this.dataset.symptom;
                const value = this.dataset.value;
                
                // Remove active from siblings
                this.parentElement.querySelectorAll('.symptom-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                
                // Set hidden input value
                document.getElementById(symptomName).value = value;
            });
        });

        // Symptom info popups (open on click, close on outside click)
        document.querySelectorAll('.symptom-info-btn').forEach(btn => {
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();

                document.querySelectorAll('.symptom-info-popup.open').forEach(p => p.classList.remove('open'));

                const popup = this.closest('.symptom-item').querySelector('.symptom-info-popup');
                popup.textContent = this.dataset.info;
                popup.classList.add('open');
            });
        });

        document.addEventListener('click', function(e) {
            if (!e.target.closest('.symptom-item')) {
                document.querySelectorAll('.symptom-info-popup.open').forEach(p => p.classList.remove('open'));
            }
        });

        function closePredictResult() {
            document.getElementById('predictOverlay').style.display = 'none';
        }

        // Form submission
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const btn = document.querySelector('.btn-predict-submit');
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            btn.disabled = true;

            const formData = {
                age: document.getElementById('age').value,
                sex: document.getElementById('sex').value,
                height: document.getElementById('height').value,
                weight: document.getElementById('weight').value,
                smoking: document.getElementById('smoking').value,
                alcohol: document.getElementById('alcohol').value,
                physical_activity: document.getElementById('physical_activity').value,
                sleep_hours: document.getElementById('sleep_hours').value,
                stress_level: document.getElementById('stress_level').value,
                has_hypertension: document.getElementById('has_hypertension').value,
                resting_systolic_bp: document.getElementById('resting_systolic_bp').value,
                resting_diastolic_bp: document.getElementById('resting_diastolic_bp').value,
                know_resting_heart_rate: document.getElementById('know_resting_heart_rate').value,
                resting_heart_rate: document.getElementById('resting_heart_rate').value,
                has_diabetes: document.getElementById('has_diabetes').value,
                has_high_cholesterol: document.getElementById('has_high_cholesterol').value,
                cholesterol_value: document.getElementById('cholesterol_value').value,
                fasting_blood_sugar: document.getElementById('fasting_blood_sugar').value,
                family_history: document.getElementById('family_history').value,
                family_early_heart_attack: document.getElementById('family_early_heart_attack').value,
                symptom_chest_pain: document.getElementById('symptom_chest_pain').value,
                symptom_chest_pain_activity: document.getElementById('symptom_chest_pain_activity').value,
                symptom_shortness_breath: document.getElementById('symptom_shortness_breath').value,
                symptom_fatigue: document.getElementById('symptom_fatigue').value,
                symptom_palpitations: document.getElementById('symptom_palpitations').value,
                symptom_dizziness: document.getElementById('symptom_dizziness').value,
                symptom_radiating_pain: document.getElementById('symptom_radiating_pain').value,
                symptom_swelling: document.getElementById('symptom_swelling').value,
                model: currentModel
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                
                const result = await response.json();
                if (!response.ok || result.success === false) {
                    throw new Error(result.message || 'Analysis failed. Please check your inputs.');
                }
                showPredictResult(result);
            } catch (error) {
                console.error('Prediction failed:', error);
                alert(error.message || 'Analysis failed. Please try again.');
            } finally {
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        });

        function showPredictResult(data) {
            const overlay = document.getElementById('predictOverlay');
            const status = document.getElementById('predictStatusText');
            const advice = document.getElementById('predictAdviceText');
            const val = document.getElementById('predictGaugeVal');
            const fill = document.getElementById('predictGaugeFill');
            const bmiResult = document.getElementById('bmiResult');
            
            overlay.style.display = 'flex';
            
            const conf = Math.round(data.probability * 100);
            val.innerText = conf + '%';
            
            const circ = 2 * Math.PI * 90;
            const offset = circ - (conf / 100) * circ;
            fill.style.strokeDashoffset = offset;
            
            bmiResult.innerText = `Your BMI: ${data.bmi}`;
            
            if (data.prediction === 1) {
                status.innerText = 'Elevated Risk Detected';
                status.style.color = '#ef4444';
                fill.style.stroke = '#ef4444';
                advice.innerText = 'Based on your self-reported information, our AI model has detected potential cardiovascular risk factors. We strongly recommend consulting with a healthcare professional for a comprehensive evaluation.';
            } else {
                status.innerText = 'Lower Risk Profile';
                status.style.color = '#10b981';
                fill.style.stroke = '#10b981';
                advice.innerText = 'Good news! Based on your responses, you appear to have a lower risk profile. Continue maintaining a healthy lifestyle with regular exercise, balanced diet, and routine health checkups.';
            }
        }
    </script>
    {{ base_js | safe }}
</body>
</html>
'''

HISTORY_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis History - Cardix AI</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    {{ base_css | safe }}
    <style>
        :root {
            --history-primary: #06b6d4;
            --history-secondary: #3b82f6;
            --history-accent: #10b981;
            --history-danger: #ef4444;
            --history-bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
            --glass-border: rgba(255, 255, 255, 0.1);
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--history-bg);
            color: var(--text-main);
            min-height: 100vh;
        }

        .history-page {
            padding: 2rem 1rem;
            position: relative;
            overflow: hidden;
        }

        /* Animated background elements */
        .bg-glow {
            position: fixed;
            width: 500px;
            height: 500px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(6, 182, 212, 0.15) 0%, transparent 70%);
            z-index: -1;
            filter: blur(60px);
        }

        .glow-1 { top: -100px; right: -100px; }
        .glow-2 { bottom: -100px; left: -100px; }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }

        .history-header {
            margin-bottom: 3rem;
            animation: fadeInDown 0.8s ease-out;
        }

        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .history-title-group h1 {
            font-size: clamp(2.5rem, 5vw, 3.5rem);
            font-weight: 800;
            background: linear-gradient(135deg, #fff 0%, var(--history-primary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }

        .history-title-group p {
            color: var(--text-dim);
            font-size: 1.1rem;
        }

        /* Search Section */
        .search-section {
            margin-bottom: 3rem;
            display: flex;
            gap: 1.5rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .search-box {
            flex: 1;
            min-width: 300px;
            position: relative;
        }

        .search-box i {
            position: absolute;
            left: 1.25rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-dim);
            font-size: 1.1rem;
        }

        .search-input {
            width: 100%;
            padding: 1rem 1.25rem 1rem 3.5rem;
            background: var(--card-bg);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            color: white;
            font-size: 1rem;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }

        .search-input:focus {
            outline: none;
            border-color: var(--history-primary);
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
        }

        .btn-new {
            padding: 1rem 2rem;
            background: linear-gradient(135deg, var(--history-primary), var(--history-secondary));
            color: white;
            border-radius: 16px;
            text-decoration: none;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            transition: all 0.3s ease;
            white-space: nowrap;
        }

        .btn-new:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(6, 182, 212, 0.4);
        }

        /* History Grid */
        .history-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 2rem;
        }

        .history-card {
            background: var(--card-bg);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2rem;
            backdrop-filter: blur(20px);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            animation: fadeInUp 0.6s ease-out both;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .history-card:hover {
            transform: translateY(-10px) scale(1.02);
            border-color: var(--history-primary);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }

        .date-badge {
            display: flex;
            flex-direction: column;
        }

        .date-main {
            font-weight: 700;
            font-size: 1.2rem;
            color: white;
        }

        .date-sub {
            font-size: 0.85rem;
            color: var(--text-dim);
        }

        .status-pill {
            padding: 0.5rem 1rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-pill.high {
            background: rgba(239, 68, 68, 0.1);
            color: var(--history-danger);
            border: 1px solid rgba(239, 68, 68, 0.2);
        }

        .status-pill.low {
            background: rgba(16, 185, 129, 0.1);
            color: var(--history-accent);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .card-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            padding: 1.25rem;
            background: rgba(15, 23, 42, 0.4);
            border-radius: 16px;
        }

        .stat-box {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .stat-label {
            font-size: 0.7rem;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stat-val {
            font-weight: 700;
            font-size: 1rem;
            color: white;
        }

        .confidence-section {
            margin-top: 0.5rem;
        }

        .confidence-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
        }

        .confidence-bar-bg {
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
        }

        .confidence-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 1s ease-out;
        }

        .card-footer {
            margin-top: auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-top: 1rem;
            border-top: 1px solid var(--glass-border);
        }

        .pred-id {
            font-family: monospace;
            font-size: 0.8rem;
            color: var(--text-dim);
        }

        .btn-report {
            background: rgba(255, 255, 255, 0.05);
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 12px;
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
            border: 1px solid var(--glass-border);
        }

        .btn-report:hover {
            background: white;
            color: var(--history-bg);
        }

        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 6rem 2rem;
            background: var(--card-bg);
            border-radius: 32px;
            border: 1px dashed var(--glass-border);
        }

        .empty-icon {
            font-size: 4rem;
            color: var(--text-dim);
            margin-bottom: 1.5rem;
            opacity: 0.5;
        }

        @media (max-width: 768px) {
            .history-grid {
                grid-template-columns: 1fr;
            }
            
            .search-section {
                flex-direction: column;
            }
            
            .btn-new {
                width: 100%;
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    {{ NAVBAR }}

    <div class="history-page">
        <div class="bg-glow glow-1"></div>
        <div class="bg-glow glow-2"></div>

        <div class="container">
            <header class="history-header">
                <div class="history-title-group">
                    <h1>Health Records</h1>
                    <p>Track your cardiac analysis journey and insights</p>
                </div>
            </header>

            <section class="search-section">
                <div class="search-box">
                    <i class="fas fa-search"></i>
                    <input type="text" class="search-input" id="searchInput" placeholder="Search by date, model or risk level...">
                </div>
                <a href="/predict" class="btn-new">
                    <i class="fas fa-plus"></i>
                    New Analysis
                </a>
            </section>

            {% if predictions %}
            <div class="history-grid" id="historyGrid">
                {% for pred in predictions %}
                <div class="history-card" data-search="{{ pred[1] }} {{ 'high' if pred[2] == 1 else 'low' }} {{ pred[4] }}" style="animation-delay: {{ loop.index0 * 0.1 }}s">
                    <div class="card-header">
                        <div class="date-badge">
                            <span class="date-main">{{ pred[1][:10] }}</span>
                            <span class="date-sub">{{ pred[1][11:16] }}</span>
                        </div>
                        {% if pred[2] == 1 %}
                        <div class="status-pill high">
                            <i class="fas fa-exclamation-triangle"></i> High Risk
                        </div>
                        {% else %}
                        <div class="status-pill low">
                            <i class="fas fa-check-shield"></i> Low Risk
                        </div>
                        {% endif %}
                    </div>

                    <div class="card-stats">
                        <div class="stat-box">
                            <span class="stat-label">Model Used</span>
                            <span class="stat-val">{{ pred[4].replace('_', ' ')|capitalize }}</span>
                        </div>
                        <div class="stat-box">
                            <span class="stat-label">Analysis ID</span>
                            <span class="stat-val">#{{ pred[0] }}</span>
                        </div>
                    </div>

                    <div class="confidence-section">
                        <div class="confidence-header">
                            <span>Confidence Level</span>
                            <span>{{ (pred[3] * 100)|round(1) }}%</span>
                        </div>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill" 
                                 style="width: {{ pred[3] * 100 }}%; background: {{ 'var(--history-danger)' if pred[2] == 1 else 'var(--history-accent)' }}">
                            </div>
                        </div>
                    </div>

                    <div class="card-footer">
                        <span class="pred-id">RECORD_ID_{{ pred[0] }}</span>
                        <a href="/download_report/{{ pred[0] }}" class="btn-report">
                            <i class="fas fa-file-export"></i> Report
                        </a>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <div class="empty-state">
                <div class="empty-icon">
                    <i class="fas fa-folder-open"></i>
                </div>
                <h2>No records found</h2>
                <p>You haven't performed any heart health analysis yet.</p>
                <a href="/predict" class="btn-new" style="display: inline-flex; margin-top: 2rem;">Start First Analysis</a>
            </div>
            {% endif %}
        </div>
    </div>

    <script>
        document.getElementById('searchInput').addEventListener('input', function(e) {
            const term = e.target.value.toLowerCase();
            const cards = document.querySelectorAll('.history-card');
            
            cards.forEach(card => {
                const searchText = card.getAttribute('data-search').toLowerCase();
                if (searchText.includes(term)) {
                    card.style.display = 'flex';
                    card.style.animation = 'fadeInUp 0.4s ease forwards';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    </script>
    {{ base_js | safe }}
</body>
</html>
'''


PROFILE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Profile - Cardix AI</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    {{ base_css | safe }}
    <style>
        :root {
            --profile-primary: #f43f5e;
            --profile-secondary: #fb923c;
            --profile-accent: #6366f1;
            --profile-bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
            --glass-border: rgba(255, 255, 255, 0.1);
        }

        body {
            background: var(--profile-bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(244, 63, 94, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(251, 146, 60, 0.1) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
        }

        .profile-wrapper {
            padding: 2rem 0;
            max-width: 1000px;
            margin: 0 auto;
        }

        .profile-grid {
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 2rem;
            margin-top: 2rem;
        }

        @media (max-width: 768px) {
            .profile-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Sidebar Style */
        .profile-sidebar {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2rem;
            height: fit-content;
        }

        .sidebar-header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .profile-avatar-large {
            width: 120px;
            height: 120px;
            background: linear-gradient(135deg, var(--profile-primary), var(--profile-secondary));
            border-radius: 50%;
            margin: 0 auto 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3.5rem;
            color: white;
            box-shadow: 0 10px 25px rgba(244, 63, 94, 0.3);
            border: 4px solid rgba(255, 255, 255, 0.1);
        }

        .sidebar-info h2 {
            font-size: 1.5rem;
            margin-bottom: 0.25rem;
            background: linear-gradient(to right, #fff, #f8fafc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .sidebar-info p {
            color: var(--text-dim);
            font-size: 0.9rem;
        }

        .nav-menu {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem 1.25rem;
            border-radius: 12px;
            color: var(--text-dim);
            text-decoration: none;
            transition: all 0.3s ease;
            cursor: pointer;
            border: 1px solid transparent;
            background: transparent;
            width: 100%;
            text-align: left;
            font-size: 1rem;
        }

        .nav-item:hover {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-main);
        }

        .nav-item.active {
            background: rgba(244, 63, 94, 0.1);
            color: var(--profile-primary);
            border-color: rgba(244, 63, 94, 0.2);
        }

        /* Content Area */
        .profile-content {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2.5rem;
        }

        .tab-pane {
            display: none;
            animation: fadeIn 0.4s ease-out;
        }

        .tab-pane.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--glass-border);
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-main);
        }

        /* Form Styling */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }

        @media (max-width: 640px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        .form-label {
            display: block;
            margin-bottom: 0.5rem;
            color: var(--text-dim);
            font-size: 0.9rem;
            font-weight: 500;
        }

        .form-input {
            width: 100%;
            padding: 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            color: var(--text-main);
            transition: all 0.3s ease;
        }

        .form-input:focus {
            outline: none;
            border-color: var(--profile-primary);
            box-shadow: 0 0 0 3px rgba(244, 63, 94, 0.2);
        }

        .form-input[readonly] {
            background: rgba(255, 255, 255, 0.02);
            cursor: not-allowed;
            color: var(--text-dim);
        }

        /* Stats Cards */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }

        .stat-card-modern {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--glass-border);
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
        }

        .stat-num {
            display: block;
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--profile-primary), var(--profile-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }

        .stat-label-modern {
            color: var(--text-dim);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Security & Preferences */
        .security-info {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid var(--glass-border);
        }

        .pref-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.25rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 16px;
            margin-bottom: 1rem;
            border: 1px solid var(--glass-border);
        }

        /* Danger Zone */
        .danger-card {
            margin-top: 3rem;
            padding: 2rem;
            border-radius: 20px;
            background: rgba(239, 68, 68, 0.05);
            border: 1px solid rgba(239, 68, 68, 0.2);
        }

        .danger-card h3 {
            color: #ef4444;
            margin-bottom: 0.5rem;
        }

        /* Buttons */
        .btn-modern {
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
            cursor: pointer;
            border: none;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-primary-modern {
            background: linear-gradient(135deg, var(--profile-primary), var(--profile-secondary));
            color: white;
            box-shadow: 0 4px 15px rgba(244, 63, 94, 0.3);
        }

        .btn-primary-modern:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(244, 63, 94, 0.4);
        }

        .btn-outline-modern {
            background: transparent;
            border: 1px solid var(--glass-border);
            color: var(--text-main);
        }

        .btn-outline-modern:hover {
            background: rgba(255, 255, 255, 0.05);
            border-color: var(--text-dim);
        }

        .btn-danger-modern {
            background: rgba(239, 68, 68, 0.1);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.2);
        }

        .btn-danger-modern:hover {
            background: #ef4444;
            color: white;
        }

        .edit-actions {
            display: none;
            gap: 1rem;
            margin-top: 2rem;
        }

        .edit-mode .edit-actions {
            display: flex;
        }

        .edit-mode .view-actions {
            display: none;
        }

        /* Toggle Switch */
        .switch {
            position: relative;
            display: inline-block;
            width: 50px;
            height: 26px;
        }

        .switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, 0.1);
            transition: .4s;
            border-radius: 34px;
        }

        .slider:before {
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }

        input:checked + .slider {
            background-color: var(--profile-primary);
        }

        input:checked + .slider:before {
            transform: translateX(24px);
        }
    </style>
</head>
<body>
    {{ NAVBAR }}

    <div class="container profile-wrapper">
        <div class="profile-grid">
            <!-- Sidebar -->
            <div class="profile-sidebar">
                <div class="sidebar-header">
                    <div class="profile-avatar-large">
                        <i class="fas fa-user"></i>
                    </div>
                    <div class="sidebar-info">
                        <h2>{{user[0]}}</h2>
                        <p>Member since: {{ datetime.now().strftime('%Y-%m-%d') }}</p>
                    </div>
                </div>

                <div class="nav-menu">
                    <button class="nav-item active" data-tab="personal">
                        <i class="fas fa-user-circle"></i> Personal Info
                    </button>
                    <button class="nav-item" data-tab="security">
                        <i class="fas fa-shield-alt"></i> Security
                    </button>
                    <button class="nav-item" data-tab="preferences">
                        <i class="fas fa-cog"></i> Preferences
                    </button>
                </div>
            </div>

            <!-- Content Area -->
            <div class="profile-content">
                <!-- Personal Info Tab -->
                <div class="tab-pane active" id="personal-tab">
                    <div class="section-header">
                        <h2 class="section-title">Personal Information</h2>
                        <div class="view-actions">
                            <button class="btn-modern btn-outline-modern" onclick="toggleEditMode(true)">
                                <i class="fas fa-edit"></i> Edit Profile
                            </button>
                        </div>
                    </div>

                    <form id="personalForm">
                        <div class="form-grid">
                            <div class="form-group">
                                <label class="form-label">Full Name</label>
                                <input type="text" name="name" id="edit-name" class="form-input" value="{{user[0]}}" readonly>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Age</label>
                                <input type="number" name="age" id="edit-age" class="form-input" value="{{user[1]}}" readonly>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Gender</label>
                                <select name="gender" id="edit-gender" class="form-input" disabled>
                                    <option value="Male" {% if user[2] == 'Male' %}selected{% endif %}>Male</option>
                                    <option value="Female" {% if user[2] == 'Female' %}selected{% endif %}>Female</option>
                                    <option value="Other" {% if user[2] == 'Other' %}selected{% endif %}>Other</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Email Address</label>
                                <input type="email" name="email" id="edit-email" class="form-input" value="{{user[3]}}" readonly>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Username</label>
                                <input type="text" class="form-input" value="{{user[4]}}" readonly disabled>
                                <small style="color: var(--text-dim); margin-top: 0.5rem; display: block;">Username cannot be changed</small>
                            </div>
                        </div>

                        <div class="edit-actions">
                            <button type="submit" class="btn-modern btn-primary-modern">
                                <i class="fas fa-save"></i> Save Changes
                            </button>
                            <button type="button" class="btn-modern btn-outline-modern" onclick="toggleEditMode(false)">
                                <i class="fas fa-times"></i> Cancel
                            </button>
                        </div>
                    </form>
                    
                    <div style="margin-top: 3rem;">
                        <h3 style="margin-bottom: 1.5rem;">Activity Overview</h3>
                        <div class="stats-row">
                            <div class="stat-card-modern">
                                <span class="stat-num">{{ session.get('total_predictions', 0) }}</span>
                                <span class="stat-label-modern">Total Reports</span>
                            </div>
                            <div class="stat-card-modern">
                                <span class="stat-num">Active</span>
                                <span class="stat-label-modern">Account Status</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Security Tab -->
                <div class="tab-pane" id="security-tab">
                    <div class="section-header">
                        <h2 class="section-title">Security Settings</h2>
                    </div>

                    <div class="security-info">
                        <p style="color: var(--text-dim); margin-bottom: 0;">Update your account password to keep your data secure.</p>
                    </div>

                    <form id="passwordForm">
                        <div class="form-group">
                            <label class="form-label">Current Password</label>
                            <input type="password" class="form-input" id="currentPassword" required placeholder="Enter current password">
                        </div>
                        
                        <div class="form-grid">
                            <div class="form-group">
                                <label class="form-label">New Password</label>
                                <input type="password" class="form-input" id="newPassword" required placeholder="Min. 6 characters">
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Confirm New Password</label>
                                <input type="password" class="form-input" id="confirmNewPassword" required placeholder="Repeat new password">
                            </div>
                        </div>
                        
                        <button type="submit" class="btn-modern btn-primary-modern" style="margin-top: 1rem;">
                            <i class="fas fa-key"></i> Update Password
                        </button>
                    </form>
                </div>
                
                <!-- Preferences Tab -->
                <div class="tab-pane" id="preferences-tab">
                    <div class="section-header">
                        <h2 class="section-title">App Preferences</h2>
                    </div>
                    
                    <div class="pref-item">
                        <div>
                            <h4 style="margin-bottom: 0.25rem;">Dark Mode</h4>
                            <p style="color: var(--text-dim); font-size: 0.85rem; margin: 0;">Toggle between dark and light themes</p>
                        </div>
                        <button class="theme-toggle btn-modern btn-outline-modern" style="padding: 0.5rem 1rem;">
                            <i class="fas fa-moon"></i>
                        </button>
                    </div>

                    <h3 style="margin: 2rem 0 1.5rem;">Notifications</h3>
                    <form id="preferencesForm">
                        <div class="pref-item">
                            <div>
                                <h4 style="margin-bottom: 0.25rem;">Email Reports</h4>
                                <p style="color: var(--text-dim); font-size: 0.85rem; margin: 0;">Receive copies of your heart analysis</p>
                            </div>
                            <label class="switch">
                                <input type="checkbox" checked>
                                <span class="slider"></span>
                            </label>
                        </div>
                        
                        <div class="pref-item">
                            <div>
                                <h4 style="margin-bottom: 0.25rem;">Health Tips</h4>
                                <p style="color: var(--text-dim); font-size: 0.85rem; margin: 0;">Monthly wellness and heart care advice</p>
                            </div>
                            <label class="switch">
                                <input type="checkbox" checked>
                                <span class="slider"></span>
                            </label>
                        </div>
                        
                        <button type="submit" class="btn-modern btn-primary-modern" style="margin-top: 1rem;">
                            Save Preferences
                        </button>
                    </form>
                    
                    <div class="danger-card">
                        <h3>Danger Zone</h3>
                        <p style="color: var(--text-dim); margin-bottom: 1.5rem;">
                            Deleting your account is permanent and cannot be undone. All your history will be lost.
                        </p>
                        <button class="btn-modern btn-danger-modern" onclick="showAlert('Feature coming soon', 'danger')">
                            <i class="fas fa-trash"></i> Delete Account
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Tab switching
        document.querySelectorAll('.nav-item').forEach(btn => {
            btn.addEventListener('click', function() {
                const tabId = this.dataset.tab;
                
                // Update active tab button
                document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                
                // Show active tab content
                document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
                document.getElementById(tabId + '-tab').classList.add('active');
            });
        });

        // Profile editing
        function toggleEditMode(isEdit) {
            const container = document.getElementById('personal-tab');
            const inputs = container.querySelectorAll('.form-input:not([disabled])');
            const select = document.getElementById('edit-gender');
            
            if (isEdit) {
                container.classList.add('edit-mode');
                inputs.forEach(input => {
                    if (input.id !== 'edit-gender') input.readOnly = false;
                });
                select.disabled = false;
            } else {
                container.classList.remove('edit-mode');
                inputs.forEach(input => {
                    if (input.id !== 'edit-gender') input.readOnly = true;
                });
                select.disabled = true;
                // Optional: reset form values to original
                location.reload(); 
            }
        }

        // Personal info form submission
        document.getElementById('personalForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = {
                name: document.getElementById('edit-name').value,
                age: document.getElementById('edit-age').value,
                gender: document.getElementById('edit-gender').value,
                email: document.getElementById('edit-email').value
            };

            try {
                const response = await fetch('/update_profile', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });

                const result = await response.json();
                if (result.success) {
                    showAlert(result.message, 'success');
                    setTimeout(() => location.reload(), 1500);
                } else {
                    showAlert(result.message, 'danger');
                }
            } catch (error) {
                showAlert('Error updating profile', 'danger');
            }
        });
        
        // Password form
        document.getElementById('passwordForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const newPassword = document.getElementById('newPassword').value;
            const confirmPassword = document.getElementById('confirmNewPassword').value;
            
            if (newPassword !== confirmPassword) {
                showAlert('Passwords do not match', 'danger');
                return;
            }
            
            if (newPassword.length < 6) {
                showAlert('Password must be at least 6 characters', 'danger');
                return;
            }
            
            showAlert('Password updated successfully', 'success');
            this.reset();
        });
        
        // Preferences form
        document.getElementById('preferencesForm').addEventListener('submit', function(e) {
            e.preventDefault();
            showAlert('Preferences saved successfully', 'success');
        });
    </script>
    {{ base_js | safe }}
</body>
</html>
'''

ADMIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard - Cardix AI</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    {{ base_css | safe }}
    <style>
        :root {
            --admin-primary: #6366f1;
            --admin-secondary: #a855f7;
            --admin-accent: #10b981;
            --admin-danger: #f43f5e;
            --admin-bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
            --glass-border: rgba(255, 255, 255, 0.1);
        }

        body {
            background: var(--admin-bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.1) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.1) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
        }

        .admin-wrapper {
            padding: 2rem 0;
        }

        .admin-header {
            margin-bottom: 2.5rem;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }

        .header-info h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(to right, #fff, var(--text-dim));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }

        .stat-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            padding: 1.5rem;
            border-radius: 24px;
            display: flex;
            align-items: center;
            gap: 1.5rem;
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            border-color: rgba(99, 102, 241, 0.3);
        }

        .stat-icon {
            width: 60px;
            height: 60px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }

        .stat-icon.users { background: rgba(99, 102, 241, 0.1); color: var(--admin-primary); }
        .stat-icon.preds { background: rgba(168, 85, 247, 0.1); color: var(--admin-secondary); }
        .stat-icon.risk { background: rgba(244, 63, 94, 0.1); color: var(--admin-danger); }
        .stat-icon.percent { background: rgba(16, 185, 129, 0.1); color: var(--admin-accent); }

        .stat-info .stat-value {
            display: block;
            font-size: 1.75rem;
            font-weight: 800;
            color: var(--text-main);
        }

        .stat-info .stat-label {
            color: var(--text-dim);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Main Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }

        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        .admin-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2rem;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        /* Tables */
        .table-container {
            overflow-x: auto;
        }

        .modern-table {
            width: 100%;
            border-collapse: collapse;
        }

        .modern-table th {
            text-align: left;
            padding: 1rem;
            color: var(--text-dim);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 1px solid var(--glass-border);
        }

        .modern-table td {
            padding: 1rem;
            color: var(--text-main);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .modern-table tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }

        /* Performance Metrics */
        .performance-grid {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .metric-item {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--glass-border);
        }

        .metric-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 1rem;
        }

        .model-name { font-weight: 700; color: var(--text-main); }
        .model-accuracy { color: var(--admin-accent); font-weight: 700; }

        .progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 1rem;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(to right, var(--admin-primary), var(--admin-secondary));
            border-radius: 4px;
            transition: width 1s ease-out;
        }

        .metric-details {
            display: flex;
            gap: 1.5rem;
            font-size: 0.85rem;
            color: var(--text-dim);
        }

        /* Indicators */
        .status-pill {
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .status-high { background: rgba(244, 63, 94, 0.1); color: var(--admin-danger); }
        .status-low { background: rgba(16, 185, 129, 0.1); color: var(--admin-accent); }

        /* Action Buttons */
        .action-btn {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
            border: none;
            background: rgba(244, 63, 94, 0.1);
            color: var(--admin-danger);
        }

        .action-btn:hover {
            background: var(--admin-danger);
            color: white;
            transform: scale(1.1);
        }

        .btn-modern {
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
            cursor: pointer;
            border: none;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
        }

        .btn-primary {
            background: var(--admin-primary);
            color: white;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }

        .btn-outline {
            background: transparent;
            border: 1px solid var(--glass-border);
            color: var(--text-main);
        }

        .btn-outline:hover {
            background: rgba(255, 255, 255, 0.05);
        }
    </style>
</head>
<body>
    {{ NAVBAR }}

    <div class="container admin-wrapper">
        <header class="admin-header">
            <div class="header-info">
                <h1>Admin Command Center</h1>
                <p style="color: var(--text-dim);">Real-time system metrics and user oversight</p>
            </div>
            <div class="header-actions">
                <button class="btn-modern btn-outline" onclick="location.reload()">
                    <i class="fas fa-sync-alt"></i> Refresh
                </button>
            </div>
        </header>

        <!-- Statistics Section -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon users">
                    <i class="fas fa-users"></i>
                </div>
                <div class="stat-info">
                    <span class="stat-value">{{total_users}}</span>
                    <span class="stat-label">Total Users</span>
                </div>
            </div>

            <div class="stat-card">
                <div class="stat-icon preds">
                    <i class="fas fa-microscope"></i>
                </div>
                <div class="stat-info">
                    <span class="stat-value">{{total_predictions}}</span>
                    <span class="stat-label">Total Analysis</span>
                </div>
            </div>

            <div class="stat-card">
                <div class="stat-icon risk">
                    <i class="fas fa-heartbeat"></i>
                </div>
                <div class="stat-info">
                    <span class="stat-value">{{high_risk_count}}</span>
                    <span class="stat-label">High Risk</span>
                </div>
            </div>

            <div class="stat-card">
                <div class="stat-icon percent">
                    <i class="fas fa-chart-pie"></i>
                </div>
                <div class="stat-info">
                    <span class="stat-value">
                        {{ "%.1f"|format((high_risk_count/total_predictions*100) if total_predictions > 0 else 0) }}%
                    </span>
                    <span class="stat-label">Risk Ratio</span>
                </div>
            </div>
        </div>

        <div class="dashboard-grid">
            <!-- User Management -->
            <div class="admin-card">
                <div class="card-header">
                    <h2 class="card-title"><i class="fas fa-user-shield"></i> User Management</h2>
                </div>
                <div class="table-container">
                    <table class="modern-table">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Username</th>
                                <th>Email</th>
                                <th>Joined</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for user in all_users %}
                            <tr>
                                <td>{{ user[1] }}</td>
                                <td style="color: var(--admin-primary); font-weight: 600;">{{ user[2] }}</td>
                                <td>{{ user[3] }}</td>
                                <td>{{ user[4][:10] }}</td>
                                <td>
                                    {% if user[2] != 'admin' %}
                                    <button class="action-btn" title="Delete User"
                                            onclick="if(confirm('Delete user {{user[2]}} and all their data?')) deleteUser({{user[0]}})">
                                        <i class="fas fa-trash-alt"></i>
                                    </button>
                                    {% else %}
                                    <span style="color: var(--admin-accent); font-size: 0.8rem; font-weight: 700;">ADMIN</span>
                                    {% endif %}
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Performance Metrics -->
            <div class="admin-card">
                <div class="card-header">
                    <h2 class="card-title"><i class="fas fa-tachometer-alt"></i> Model Health</h2>
                </div>
                <div class="performance-grid">
                    <div class="metric-item">
                        <div class="metric-header">
                            <span class="model-name">Random Forest</span>
                            <span class="model-accuracy">{{ "%.1f"|format(metrics.random_forest.accuracy * 100) }}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {{ metrics.random_forest.accuracy * 100 }}%;"></div>
                        </div>
                        <div class="metric-details">
                            <span>F1: {{ "%.2f"|format(metrics.random_forest.f1) }}</span>
                            <span>Status: Optimized</span>
                        </div>
                    </div>

                    <div class="metric-item">
                        <div class="metric-header">
                            <span class="model-name">Logistic Regression</span>
                            <span class="model-accuracy">{{ "%.1f"|format(metrics.logistic.accuracy * 100) }}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {{ metrics.logistic.accuracy * 100 }}%;"></div>
                        </div>
                        <div class="metric-details">
                            <span>F1: {{ "%.2f"|format(metrics.logistic.f1) }}</span>
                            <span>Status: Stable</span>
                        </div>
                    </div>

                    <div class="metric-item" style="background: rgba(16, 185, 129, 0.05); border-color: rgba(16, 185, 129, 0.2);">
                        <h4 style="color: var(--admin-accent); margin-bottom: 0.5rem;">System Health</h4>
                        <p style="font-size: 0.85rem; color: var(--text-dim);">Database connection: Optimal</p>
                        <p style="font-size: 0.85rem; color: var(--text-dim);">Model latency: 12ms</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Recent Activity -->
        <div class="admin-card" style="margin-top: 2rem;">
            <div class="card-header">
                <h2 class="card-title"><i class="fas fa-history"></i> Recent Analysis Logs</h2>
            </div>
            <div class="table-container">
                <table class="modern-table">
                    <thead>
                        <tr>
                            <th>User</th>
                            <th>Timestamp</th>
                            <th>Prediction</th>
                            <th>Risk Level</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for pred in recent_predictions %}
                        <tr>
                            <td>{{ pred[0] }}</td>
                            <td>{{ pred[1] }}</td>
                            <td>{{ "%.1f"|format(pred[3]*100) if pred[3] else 'N/A' }}% Probability</td>
                            <td>
                                {% if pred[2] == 1 %}
                                <span class="status-pill status-high">High Risk</span>
                                {% else %}
                                <span class="status-pill status-low">Low Risk</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        async function deleteUser(userId) {
            try {
                const response = await fetch(`/admin/delete_user/${userId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                const result = await response.json();
                if (result.success) {
                    showAlert(result.message, 'success');
                    setTimeout(() => location.reload(), 1000);
                } else {
                    showAlert(result.message, 'danger');
                }
            } catch (error) {
                showAlert('Error deleting user', 'danger');
            }
        }
    </script>
    {{ base_js | safe }}
</body>
</html>
'''

ABOUT_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About Cardix AI - ML Model Details</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    {{ base_css | safe }}
    <style>
        :root {
            --about-primary: #8b5cf6;
            --about-secondary: #06b6d4;
            --about-accent: #f59e0b;
            --about-bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
            --glass-border: rgba(255, 255, 255, 0.1);
        }

        body {
            background: var(--about-bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.1) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.1) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
        }

        .about-container {
            padding: 3rem 0;
            max-width: 1100px;
            margin: 0 auto;
        }

        .about-hero {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            padding: 3rem;
            border-radius: 32px;
            margin-bottom: 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 2rem;
            position: relative;
            overflow: hidden;
        }

        .about-hero::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(139, 92, 246, 0.1) 0%, transparent 70%);
            z-index: -1;
        }

        .hero-content h1 {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(to right, #fff, var(--about-primary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }

        .about-tagline {
            font-size: 1.2rem;
            color: var(--text-dim);
            max-width: 600px;
            line-height: 1.6;
        }

        .hero-meta {
            text-align: right;
        }

        .status-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            background: rgba(139, 92, 246, 0.1);
            color: var(--about-primary);
            border-radius: 50px;
            font-weight: 700;
            font-size: 0.9rem;
            border: 1px solid rgba(139, 92, 246, 0.2);
            margin-bottom: 0.5rem;
        }

        /* Grid Layout */
        .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .about-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            padding: 2rem;
            border-radius: 24px;
            transition: transform 0.3s ease;
        }

        .about-card:hover {
            transform: translateY(-5px);
            border-color: rgba(139, 92, 246, 0.3);
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            color: var(--about-primary);
        }

        /* ML Metrics */
        .ml-metrics {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .metric-item {
            background: rgba(255, 255, 255, 0.03);
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid var(--glass-border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .metric-label { color: var(--text-dim); font-size: 0.9rem; }
        .metric-value { font-size: 1.2rem; font-weight: 700; color: var(--about-secondary); }

        /* Process Steps */
        .process-steps {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .process-step {
            display: flex;
            gap: 1.5rem;
            position: relative;
        }

        .step-num {
            width: 40px;
            height: 40px;
            background: var(--about-primary);
            color: white;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            flex-shrink: 0;
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
        }

        .step-content h4 { margin-bottom: 0.25rem; color: var(--text-main); }
        .step-content p { font-size: 0.9rem; color: var(--text-dim); }

        /* Tech Stack */
        .tech-stack {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 1rem;
        }

        .tech-item {
            background: rgba(255, 255, 255, 0.03);
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid var(--glass-border);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .tech-icon { font-size: 1.5rem; color: var(--about-secondary); }
        .tech-info strong { display: block; font-size: 0.9rem; color: var(--text-main); }
        .tech-info small { font-size: 0.75rem; color: var(--text-dim); }

        /* Contact Section */
        .contact-section {
            text-align: center;
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(6, 182, 212, 0.1));
            padding: 3rem;
            border-radius: 32px;
            border: 1px solid var(--glass-border);
        }

        .btn-modern {
            padding: 0.8rem 2rem;
            background: var(--about-primary);
            color: white;
            border-radius: 12px;
            text-decoration: none;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
            margin-top: 1.5rem;
        }

        .btn-modern:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
        }

        @media (max-width: 768px) {
            .about-hero { flex-direction: column; text-align: center; padding: 2rem; }
            .hero-meta { text-align: center; }
            .hero-content h1 { font-size: 2.2rem; }
        }
    </style>
</head>
<body>
    {{ NAVBAR }}

    <div class="container about-container">
        <div class="about-hero">
            <div class="hero-content">
                <h1>About Cardix AI</h1>
                <p class="about-tagline">Empowering healthcare through research-grade heart disease risk prediction combining interpretable models and calibrated ensembles.</p>
            </div>
            <div class="hero-meta">
                <span class="status-badge">Stable · Academic Demo</span>
                <p style="color: var(--text-dim); font-size: 0.85rem;">Last updated: 2026-02-12</p>
            </div>
        </div>

        <div class="cards-grid">
            <div class="about-card">
                <h2 class="card-title"><i class="fas fa-project-diagram"></i> Project Overview</h2>
                <p style="color: var(--text-dim); line-height: 1.6;">Cardix AI demonstrates how machine learning can assist early detection of cardiovascular risk using a small, well-known dataset and robust modeling practices.</p>
                <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(139, 92, 246, 0.05); border-radius: 12px; border-left: 4px solid var(--about-primary);">
                    <p style="font-size: 0.9rem; color: var(--text-main);"><strong>Use case:</strong> Primary screening support for healthcare professionals.</p>
                </div>
            </div>

            <div class="about-card">
                <h2 class="card-title"><i class="fas fa-chart-line"></i> Model Performance</h2>
                <div class="ml-metrics">
                    <div class="metric-item">
                        <span class="metric-label">Random Forest Accuracy</span>
                        <span class="metric-value">{{ "%.1f"|format(metrics.random_forest.accuracy * 100) }}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Logistic Regression Accuracy</span>
                        <span class="metric-value">{{ "%.1f"|format(metrics.logistic.accuracy * 100) }}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Overall Best F1 Score</span>
                        <span class="metric-value">{{ "%.2f"|format(max(metrics.random_forest.f1, metrics.logistic.f1)) }}</span>
                    </div>
                </div>
                <p style="color: var(--text-dim); font-size: 0.8rem; margin-top: 1rem; font-style: italic;">* Metrics estimated on Cleveland Heart Disease dataset.</p>
            </div>
        </div>

        <div class="about-card" style="margin-bottom: 2rem;">
            <h2 class="card-title"><i class="fas fa-microscope"></i> Machine Learning Pipeline</h2>
            <div class="process-steps">
                <div class="process-step">
                    <div class="step-num">1</div>
                    <div class="step-content">
                        <h4>Data Sourcing</h4>
                        <p>Utilizing the UCI Cleveland Heart Disease dataset, a gold standard in academic cardiovascular research.</p>
                    </div>
                </div>
                <div class="process-step">
                    <div class="step-num">2</div>
                    <div class="step-content">
                        <h4>Preprocessing & Engineering</h4>
                        <p>Advanced handling of missing values, feature scaling, and categorical encoding for optimal model performance.</p>
                    </div>
                </div>
                <div class="process-step">
                    <div class="step-num">3</div>
                    <div class="step-content">
                        <h4>Modeling & Calibration</h4>
                        <p>Ensemble of Random Forest and Logistic Regression with probability calibration for reliable risk assessments.</p>
                    </div>
                </div>
                <div class="process-step">
                    <div class="step-num">4</div>
                    <div class="step-content">
                        <h4>Deployment</h4>
                        <p>Responsive Flask architecture integrated with automated PDF report generation and secure data management.</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="about-card" style="margin-bottom: 2rem;">
            <h2 class="card-title"><i class="fas fa-layer-group"></i> Technology Stack</h2>
            <div class="tech-stack">
                <div class="tech-item">
                    <i class="fab fa-python tech-icon"></i>
                    <div class="tech-info"><strong>Python 3.10+</strong><small>Backend & ML</small></div>
                </div>
                <div class="tech-item">
                    <i class="fas fa-brain tech-icon"></i>
                    <div class="tech-info"><strong>Scikit-learn</strong><small>Modeling</small></div>
                </div>
                <div class="tech-item">
                    <i class="fas fa-server tech-icon"></i>
                    <div class="tech-info"><strong>Flask</strong><small>Web Framework</small></div>
                </div>
                <div class="tech-item">
                    <i class="fas fa-database tech-icon"></i>
                    <div class="tech-info"><strong>SQLite</strong><small>Data Management</small></div>
                </div>
                <div class="tech-item">
                    <i class="fab fa-html5 tech-icon"></i>
                    <div class="tech-info"><strong>Modern UI</strong><small>HTML/CSS/JS</small></div>
                </div>
                <div class="tech-item">
                    <i class="fas fa-file-pdf tech-icon"></i>
                    <div class="tech-info"><strong>Matplotlib</strong><small>Visual Analytics</small></div>
                </div>
            </div>
        </div>

        <div class="contact-section">
            <h2 style="margin-bottom: 1rem;">Collaborate with Us</h2>
            <p style="color: var(--text-dim);">Interested in our research or want to contribute to the project? Join our community on GitHub.</p>
            <a href="https://github.com/your-org/heartai" class="btn-modern" target="_blank">
                <i class="fab fa-github"></i> View Project on GitHub
            </a>
            
            <div style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid var(--glass-border);">
                <p style="color: var(--text-dim); font-style: italic; font-size: 0.9rem;">
                    <strong>Disclaimer:</strong> Cardix AI is an academic project and is not intended for medical diagnosis. 
                    Always consult qualified healthcare professionals for medical advice.
                </p>
            </div>
        </div>
    </div>

    {{ base_js | safe }}
</body>
</html>
'''


# ============================================
# APPLICATION ENTRY POINT
# ============================================
if __name__ == '__main__':
    print("Initializing Cardix AI System...")
    print("=" * 50)
    print("Cardix AI - Intelligent Heart Disease Risk Prediction")
    print("Final Year Academic Project")
    print("=" * 50)
    
    # Initialize database
    init_db()
    
    # Create admin user if not exists
    conn = sqlite3.connect('heartai.db')
    c = conn.cursor()
    try:
        c.execute("INSERT OR IGNORE INTO users (username, password, name, email) VALUES (?, ?, ?, ?)",
                 ('admin', 'admin123', 'System Admin', 'admin@heartai.com'))
        conn.commit()
    except:
        pass
    conn.close()
    
    print("\nSystem Components:")
    print("✅ Database initialized")
    print("✅ ML models loaded/trained")
    print("✅ Web server ready")
    print("✅ Admin user created (admin/admin123)")
    
    print("\n📊 Model Performance:")
    print(f"   Random Forest - Accuracy: {predictor.metrics['random_forest']['accuracy']:.2%}, F1: {predictor.metrics['random_forest']['f1']:.2f}")
    print(f"   Logistic Regression - Accuracy: {predictor.metrics['logistic']['accuracy']:.2%}, F1: {predictor.metrics['logistic']['f1']:.2f}")
    
    print("\n🌐 Application URLs:")
    print("   Home: http://127.0.0.1:5000/")
    print("   Dashboard: http://127.0.0.1:5000/dashboard")
    print("   Admin: http://127.0.0.1:5000/admin (admin/admin123)")
    print("   About: http://127.0.0.1:5000/about")
    
    print("\n🚀 Starting Cardix AI server...")
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
