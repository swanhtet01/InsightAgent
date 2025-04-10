
from typing import Any, Dict, List
import polars as pl
import openai
import json
import numpy as np
from uuid import uuid4
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def fix_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename(lambda x: x.strip().replace(" ", "_").lower())

def missing_report(df: pl.DataFrame) -> Dict[str, float]:
    return {col: df[col].null_count() / df.height for col in df.columns if df[col].null_count() > 0}

def generate_summary(df: pl.DataFrame, openai_api_key: str) -> str:
    openai.api_key = openai_api_key
    preview = df.head(10).write_csv()
    prompt = f"""You are a data analyst AI. Summarize this dataset for a business audience in:
- 3 bullet points (short)
- Executive summary (2–3 sentences)
Sample data:
{preview}"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def strategy_recommender(insights: str, openai_api_key: str) -> str:
    openai.api_key = openai_api_key
    prompt = f"""Based on these insights:

{insights}

Give 3 strategic business recommendations."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def domain_expert(df: pl.DataFrame, openai_api_key: str) -> str:
    openai.api_key = openai_api_key
    preview = df.head(10).write_csv()
    prompt = f"""This is a dataset:

{preview}

Identify the domain (e.g. sports, finance, sales), and explain what this dataset is probably about."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def suggest_best_target(df: pl.DataFrame) -> str:
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Int64]]
    if not numeric_cols:
        return ""
    return max(numeric_cols, key=lambda c: df[c].null_count() == 0)

def run_prediction(df: pl.DataFrame, target_col: str) -> Dict[str, Any]:
    df_pd = df.to_pandas().dropna()
    if target_col not in df_pd.columns:
        return {"error": "Target not found"}
    X = df_pd.drop(columns=[target_col])
    y = df_pd[target_col]
    X = pd.get_dummies(X)
    if len(X) < 5:
        return {"error": "Insufficient data for training."}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor().fit(X_train, y_train)
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    return {
        "model": "RandomForest",
        "score": score,
        "predictions": preds.tolist()[:5],
        "target": target_col
    }
