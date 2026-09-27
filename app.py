from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import os
import sqlite3
import numpy as np
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from cryptography.fernet import Fernet

from utils import (
    read_uploaded_file,
    prepare_uploaded_dataframe,
    get_risk_level,
    save_bar_chart,
    save_probability_histogram,
    save_risk_pie_chart
)

app = Flask(__name__)
app.secret_key = "change_this_secret_key_for_security"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "users.db")
KEY_PATH = os.path.join(BASE_DIR, "secret.key")
LOG_PATH = os.path.join(BASE_DIR, "encrypted_activity_log.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "txt"}


# -----------------------------
# Encryption functions
# -----------------------------

def load_or_create_key():
    if not os.path.exists(KEY_PATH):
        key = Fernet.generate_key()
        with open(KEY_PATH, "wb") as f:
            f.write(key)

    with open(KEY_PATH, "rb") as f:
        return f.read()


fernet = Fernet(load_or_create_key())


def encrypt_text(text):
    return fernet.encrypt(str(text).encode()).decode()


def write_encrypted_log(username, action, content):
    encrypted_content = encrypt_text(content[:500])
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"User: {username} | Action: {action} | Data: {encrypted_content}\n")


# -----------------------------
# File safety
# -----------------------------

def allowed_file(filename):
    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


# -----------------------------
# Database
# -----------------------------

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    existing_admin = conn.execute(
        "SELECT * FROM users WHERE username = ?",
        ("admin",)
    ).fetchone()

    if existing_admin is None:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", generate_password_hash("Admin@123"), "admin")
        )

    conn.commit()
    conn.close()


def get_current_user():
    if "user_id" not in session:
        return None

    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (session["user_id"],)
    ).fetchone()
    conn.close()

    return user


# -----------------------------
# Decorators
# -----------------------------

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = get_current_user()

        if user is None or user["role"] != "admin":
            flash("Admin access required.")
            return redirect(url_for("home"))

        return f(*args, **kwargs)
    return wrapper


# -----------------------------
# Load ML models
# -----------------------------

lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_model.pkl"))
rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

bert_available = False
bert_predictor = None

try:
    from bert_utils import BertPredictor

    bert_path = os.path.join(MODELS_DIR, "bert_model")

    if os.path.exists(bert_path):
        bert_predictor = BertPredictor(bert_path)
        bert_available = True
        print("BERT loaded successfully.")
    else:
        print("BERT folder not found:", bert_path)

except Exception as e:
    print("BERT loading failed:", e)
    bert_available = False


# -----------------------------
# Authentication
# -----------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db_connection()

        user = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        ).fetchone()

        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            return redirect(url_for("home"))

        flash("Invalid username or password.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# -----------------------------
# Home
# -----------------------------

@app.route("/")
@login_required
def home():
    current_user = get_current_user()

    performance = {
        "Logistic Regression": {"accuracy": 0.98, "precision": 0.98, "recall": 0.98},
        "Random Forest": {"accuracy": 0.99, "precision": 0.99, "recall": 0.99},
        "SVM": {"accuracy": 0.99, "precision": 0.99, "recall": 0.99},
        "BERT": {"accuracy": 0.95, "precision": 0.95, "recall": 0.95} if bert_available else None
    }

    model_status = {
        "Logistic Regression": "Ready",
        "Random Forest": "Ready",
        "SVM": "Ready",
        "BERT": "Ready" if bert_available else "Not Loaded"
    }

    return render_template(
        "index.html",
        current_user=current_user,
        performance=performance,
        model_status=model_status,
        bert_available=bert_available
    )


# -----------------------------
# Prediction engine
# -----------------------------

def run_prediction_on_dataframe(df):
    df, text_col = prepare_uploaded_dataframe(df)

    X = df["clean_text"]
    X_tfidf = tfidf.transform(X)

    lr_preds = lr_model.predict(X_tfidf)
    rf_preds = rf_model.predict(X_tfidf)
    svm_preds = svm_model.predict(X_tfidf)

    lr_probs = lr_model.predict_proba(X_tfidf)[:, 1] if hasattr(lr_model, "predict_proba") else [0.5] * len(df)
    rf_probs = rf_model.predict_proba(X_tfidf)[:, 1] if hasattr(rf_model, "predict_proba") else [0.5] * len(df)

    if hasattr(svm_model, "decision_function"):
        svm_scores = svm_model.decision_function(X_tfidf)
        svm_probs = 1 / (1 + np.exp(-svm_scores))
    else:
        svm_probs = [0.5] * len(df)

    runtime_bert_available = False
    bert_preds = None
    bert_probs = None

    if bert_available:
        try:
            bert_preds, bert_probs = bert_predictor.predict_batch(X.tolist())
            runtime_bert_available = True
        except Exception as e:
            print("BERT prediction failed:", e)
            runtime_bert_available = False

    ensemble_preds = []
    risk_levels = []

    for i in range(len(df)):
        vote_list = [
            int(lr_preds[i]),
            int(rf_preds[i]),
            int(svm_preds[i])
        ]

        prob_list = [
            float(lr_probs[i]),
            float(rf_probs[i]),
            float(svm_probs[i])
        ]

        if runtime_bert_available:
            vote_list.append(int(bert_preds[i]))
            prob_list.append(float(bert_probs[i]))

        threshold = 3 if runtime_bert_available else 2
        final_pred = 1 if sum(vote_list) >= threshold else 0

        avg_prob = sum(prob_list) / len(prob_list)

        ensemble_preds.append(final_pred)
        risk_levels.append(get_risk_level(avg_prob))

    df["LR Prediction"] = ["Phishing" if x == 1 else "Legitimate" for x in lr_preds]
    df["LR Probability"] = [round(float(x), 3) for x in lr_probs]

    df["RF Prediction"] = ["Phishing" if x == 1 else "Legitimate" for x in rf_preds]
    df["RF Probability"] = [round(float(x), 3) for x in rf_probs]

    df["SVM Prediction"] = ["Phishing" if x == 1 else "Legitimate" for x in svm_preds]
    df["SVM Probability"] = [round(float(x), 3) for x in svm_probs]

    if runtime_bert_available:
        df["BERT Prediction"] = ["Phishing" if x == 1 else "Legitimate" for x in bert_preds]
        df["BERT Probability"] = [round(float(x), 3) for x in bert_probs]

    df["Final Prediction"] = ["Phishing" if x == 1 else "Legitimate" for x in ensemble_preds]
    df["Risk Level"] = risk_levels

    lr_count = int(sum(lr_preds))
    rf_count = int(sum(rf_preds))
    svm_count = int(sum(svm_preds))
    bert_count = int(sum(bert_preds)) if runtime_bert_available else None
    final_count = int(sum(ensemble_preds))
    high_risk_count = int(sum(1 for x in risk_levels if x == "HIGH"))

    model_counts = {
        "LR": lr_count,
        "RF": rf_count,
        "SVM": svm_count,
        "Final": final_count
    }

    if runtime_bert_available:
        model_counts["BERT"] = bert_count

    save_bar_chart(
        model_counts,
        os.path.join(OUTPUT_DIR, "attacks_by_model.png")
    )

    save_probability_histogram(
        lr_probs,
        rf_probs,
        os.path.join(OUTPUT_DIR, "probability_distribution.png")
    )

    save_risk_pie_chart(
        risk_levels,
        os.path.join(OUTPUT_DIR, "risk_distribution.png")
    )

    return {
        "df": df,
        "text_col": text_col,
        "total_rows": len(df),
        "lr_count": lr_count,
        "rf_count": rf_count,
        "svm_count": svm_count,
        "bert_count": bert_count,
        "final_count": final_count,
        "high_risk_count": high_risk_count,
        "bert_available": runtime_bert_available
    }


# -----------------------------
# File prediction
# -----------------------------

@app.route("/predict-file", methods=["POST"])
@login_required
def predict_file():
    if "file" not in request.files:
        flash("No file uploaded.")
        return redirect(url_for("home"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("home"))

    original_filename = file.filename
    safe_filename = secure_filename(original_filename)

    if not allowed_file(safe_filename):
        flash("Unsupported file type. Upload CSV, Excel, or TXT only.")
        return redirect(url_for("home"))

    safe_path = os.path.join(UPLOAD_DIR, safe_filename)
    file.save(safe_path)

    try:
        with open(safe_path, "rb") as saved_file:
            class FileWrapper:
                def __init__(self, path, filename):
                    self.path = path
                    self.filename = filename

                def read(self):
                    with open(self.path, "rb") as f:
                        return f.read()

            wrapped_file = FileWrapper(safe_path, safe_filename)
            df = read_uploaded_file(wrapped_file)

        result = run_prediction_on_dataframe(df)

        user = get_current_user()
        write_encrypted_log(
            user["username"],
            "file_prediction",
            f"Uploaded file: {safe_filename}, rows: {result['total_rows']}"
        )

    except Exception as e:
        flash(str(e))
        return redirect(url_for("home"))

    preview_rows = result["df"].head(50).to_dict(orient="records")

    return render_template(
        "result.html",
        current_user=get_current_user(),
        input_type="File Upload",
        total_rows=result["total_rows"],
        lr_count=result["lr_count"],
        rf_count=result["rf_count"],
        svm_count=result["svm_count"],
        bert_count=result["bert_count"],
        final_count=result["final_count"],
        high_risk_count=result["high_risk_count"],
        preview_rows=preview_rows,
        bert_available=result["bert_available"]
    )


# -----------------------------
# Text prediction
# -----------------------------

@app.route("/predict-text", methods=["POST"])
@login_required
def predict_text():
    email_text = request.form.get("email_text", "").strip()

    if not email_text:
        flash("Please paste text before analysis.")
        return redirect(url_for("home"))

    df = pd.DataFrame({"text": [email_text]})
    result = run_prediction_on_dataframe(df)

    user = get_current_user()
    write_encrypted_log(
        user["username"],
        "text_prediction",
        email_text
    )

    row = result["df"].iloc[0].to_dict()

    return render_template(
        "text_result.html",
        current_user=get_current_user(),
        input_text=email_text,
        row=row,
        bert_available=result["bert_available"]
    )


# -----------------------------
# Admin user management
# -----------------------------

@app.route("/admin/users", methods=["GET", "POST"])
@login_required
@admin_required
def admin_users():
    current_user = get_current_user()

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        role = request.form.get("role", "user").strip()

        if not username or not password:
            flash("Username and password are required.")
            return redirect(url_for("admin_users"))

        if role not in ["admin", "user"]:
            flash("Invalid role.")
            return redirect(url_for("admin_users"))

        conn = get_db_connection()

        existing = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        ).fetchone()

        if existing:
            conn.close()
            flash("Username already exists.")
            return redirect(url_for("admin_users"))

        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), role)
        )

        conn.commit()
        conn.close()

        flash("User created successfully.")
        return redirect(url_for("admin_users"))

    conn = get_db_connection()
    users = conn.execute(
        "SELECT id, username, role, created_at FROM users ORDER BY id ASC"
    ).fetchall()
    conn.close()

    return render_template(
        "admin_users.html",
        users=users,
        current_user=current_user
    )


@app.route("/admin/delete-user/<int:user_id>", methods=["POST"])
@login_required
@admin_required
def delete_user(user_id):
    current_user = get_current_user()

    conn = get_db_connection()

    user = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()

    if user is None:
        flash("User not found.")

    elif user["username"] == "admin":
        flash("Default admin cannot be deleted.")

    elif user["id"] == current_user["id"]:
        flash("You cannot delete your own account.")

    else:
        conn.execute(
            "DELETE FROM users WHERE id = ?",
            (user_id,)
        )
        conn.commit()
        flash("User deleted successfully.")

    conn.close()
    return redirect(url_for("admin_users"))


# -----------------------------
# Run app
# -----------------------------

if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000, use_reloader=False)