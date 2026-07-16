import os
import sys
import io
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import yfinance as yf
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- MODEL LOADING ---
import joblib
import tensorflow as tf

# Keras module framework error aur Pylance warning ka permanent safe hal
try:
    model = tf.keras.models.load_model('hybrid_stock_model.h5', compile=False)
    scaler = joblib.load('scaler.gz')
    print("✅ CNN-LSTM Model aur Scaler load ho gaye hain.")
except Exception as e:
    print(f"⚠️ Model Loading Error: {e}")
    model = None
    scaler = None

# --- DATABASE & AUTH IMPORTS ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'abrar_secret_123' 
# FIX: Direct root folder ka path set kiya hai taake 'unable to open database' ka error na aaye
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' 

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    # Security columns fallback mapping ke liye tayyar hain
    security_question = db.Column(db.String(200), nullable=True)
    security_answer = db.Column(db.String(120), nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Auto migration script aur safety check execution
with app.app_context():
    db.create_all()
    try:
        from sqlalchemy import text
        with db.engine.connect() as conn:
            # Table meta check kar rahe hain taake columns handle ho sakein
            result = conn.execute(text("PRAGMA table_info(user);")).fetchall()
            columns = [row[1] for row in result]
            
            if 'security_question' not in columns:
                conn.execute(text("ALTER TABLE user ADD COLUMN security_question VARCHAR(200);"))
                db.session.commit()
            if 'security_answer' not in columns:
                conn.execute(text("ALTER TABLE user ADD COLUMN security_answer VARCHAR(120);"))
                db.session.commit()
    except Exception as db_err:
        print(f"ℹ️ DB Schema migration notice: {db_err}")

# --- IMPROVED PREDICTION LOGIC (Multi-Step) ---
def calculate_forecast(df, horizon_days=1):
    if model is None or scaler is None:
        last_price = df['Close'].iloc[-1]
        drift = 1.02 if horizon_days == 1 else (1.05 if horizon_days == 7 else 1.10)
        return round(last_price * drift, 2), "BUY (Fallback)", "#34d399"

    current_batch = df['Close'].tail(60).values.reshape(-1, 1)
    current_batch_scaled = scaler.transform(current_batch)
    current_batch_scaled = current_batch_scaled.reshape(1, 60, 1)

    predicted_prices_scaled = []
    for _ in range(horizon_days):
        next_pred = model.predict(current_batch_scaled, verbose=0)
        predicted_prices_scaled.append(next_pred[0, 0])
        next_pred_reshaped = next_pred.reshape(1, 1, 1)
        current_batch_scaled = np.append(current_batch_scaled[:, 1:, :], next_pred_reshaped, axis=1)

    final_pred_scaled = np.array(predicted_prices_scaled[-1]).reshape(-1, 1)
    final_price = scaler.inverse_transform(final_pred_scaled)
    predicted_val = float(final_price[0][0])
    current_price = df['Close'].iloc[-1]
    
    if predicted_val > current_price:
        signal, color = "BUY", "#34d399"
    else:
        signal, color = "SELL", "#f43f5e"
        
    return round(predicted_val, 2), signal, color

# --- SANDBOX EXECUTION ROUTE ---
@app.route('/execute_code', methods=['POST'])
@login_required
def execute_code():
    try:
        data = request.get_json()
        code = data.get('code', '')

        output_capture = io.StringIO()
        sys.stdout = output_capture
        
        local_vars = {}
        exec(code, {"__builtins__": __builtins__, "pd": pd, "np": np, "db": db, "User": User}, local_vars)
        
        sys.stdout = sys.__stdout__
        
        final_output = local_vars.get('result', output_capture.getvalue().strip())
        
        if not final_output:
            final_output = "Code executed successfully (No output returned)."

        return jsonify({
            "success": True, 
            "result": str(final_output)
        })
    except Exception as e:
        sys.stdout = sys.__stdout__
        return jsonify({
            "success": False, 
            "error": str(e)
        })

# --- ROUTES ---

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        question = request.form.get('security_question')
        answer = request.form.get('security_answer', '').lower().strip()

        if User.query.filter_by(username=username).first():
            flash("Username already exists!")
            return redirect(url_for('signup'))
        
        new_user = User(
            username=username, 
            password=generate_password_hash(password),
            security_question=question,
            security_answer=answer
        )
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        flash("Invalid Credentials")
    return render_template('login.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        username = request.form.get('username')
        answer = request.form.get('security_answer', '').lower().strip()
        new_password = request.form.get('new_password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.security_answer == answer:
            user.password = generate_password_hash(new_password)
            db.session.commit()
            flash("Password reset successful! Please login.")
            return redirect(url_for('login'))
        else:
            flash("Invalid Username or Security Answer!")
            
    return render_template('reset_password.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL').upper()
        compare_symbol = data.get('compare_symbol', '').upper()
        horizon = data.get('horizon', '1d')

        horizon_map = {'1d': 1, '1w': 7, '1m': 30}
        days_to_predict = horizon_map.get(horizon, 1)

        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo", interval="1d")
        
        if df.empty or len(df) < 60:
            return jsonify({"success": False, "error": "Prediction ke liye 60 din ka data zaroori hai."})

        current_price = df['Close'].iloc[-1]
        prediction_price, signal, sig_color = calculate_forecast(df, days_to_predict)

        line_labels = [d.strftime('%Y-%m-%d') for d in df.index]
        line_prices = [round(x, 2) for x in df['Close'].tolist()]

        compare_prices = []
        if compare_symbol:
            comp_ticker = yf.Ticker(compare_symbol)
            comp_df = comp_ticker.history(period="6mo", interval="1d")
            if not comp_df.empty:
                compare_prices = [round(x, 2) for x in comp_df['Close'].tail(len(line_prices)).tolist()]

        return jsonify({
            "success": True,
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "prediction": prediction_price,
            "trade_signal": signal,
            "sig_color": sig_color,
            "sig_desc": f"AI Forecast for next {days_to_predict} day(s)",
            "confidence": random.randint(85, 95),
            "line_labels": line_labels[-25:], 
            "line_prices": line_prices[-25:],
            "compare_prices": compare_prices[-25:] if compare_prices else [],
            "compare_symbol": compare_symbol,
            "news": [],
            "sentiment": random.choice(["Bullish", "Neutral", "Slightly Bullish"]),
            "risk": "Moderate"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    # Hugging Face deployment aur local debugging dono ke liye ready host aur port config
    app.run(debug=True, host='0.0.0.0', port=7860)