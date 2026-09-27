import matplotlib
matplotlib.use("Agg")

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_uploaded_file(file):
    filename = file.filename.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(file)

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(file)

    if filename.endswith(".txt"):
        content = file.read().decode("utf-8", errors="ignore")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) == 0:
            lines = [content]
        return pd.DataFrame({"text": lines})

    raise ValueError("Unsupported file format. Please upload CSV, Excel, or TXT file.")


def prepare_uploaded_dataframe(df):
    df = df.copy()

    possible_text_cols = [
        "text",
        "email",
        "email_text",
        "message",
        "Email Text",
        "text_combined",
        "content",
        "body",
        "mail"
    ]

    text_col = None

    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if object_cols:
            text_col = object_cols[0]
        else:
            text_col = df.columns[0]

    df["clean_text"] = df[text_col].astype(str).apply(clean_text)
    return df, text_col


def get_risk_level(prob):
    if prob >= 0.80:
        return "HIGH"
    elif prob >= 0.50:
        return "MEDIUM"
    elif prob >= 0.20:
        return "LOW"
    return "NORMAL"


def save_bar_chart(model_counts, output_path):
    plt.figure(figsize=(8, 5))
    plt.bar(model_counts.keys(), model_counts.values())
    plt.title("Attack Detections by Model")
    plt.ylabel("Number of Attack Detections")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_probability_histogram(lr_probs, rf_probs, output_path):
    plt.figure(figsize=(8, 5))
    plt.hist(lr_probs, bins=15, alpha=0.6, label="Logistic Regression")
    plt.hist(rf_probs, bins=15, alpha=0.6, label="Random Forest")
    plt.title("Attack Probability Distribution")
    plt.xlabel("Attack Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_risk_pie_chart(risk_levels, output_path):
    counts = pd.Series(risk_levels).value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    plt.title("Risk Category Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()