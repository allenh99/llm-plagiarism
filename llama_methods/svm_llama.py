from datasets import load_dataset
import pandas as pd
import requests
import json
import numpy as np
import os

dataset = load_dataset("Hello-SimpleAI/HC3", "all", split="train")

df = dataset.to_pandas()
df = df[df["chatgpt_answers"].notnull()]
df = df[df["source"] != "wikipedia"]

df = df.reset_index(drop=True)

rows = []
for _, row in df.iterrows():
    rows.append({
        "id": row["id"],
        "source": row["source"],
        "question": row["question"],
        "answer": row["human_answers"],
        "answer_source": 0 
    })
    rows.append({
        "id": row["id"],
        "source": row["source"],
        "question": row["question"],
        "answer": row["chatgpt_answers"],
        "answer_source": 1
    })

df_flat = pd.DataFrame(rows)

def get_llama3_embedding(text):
    try:
        response = requests.post(
            'http://localhost:11434/api/embeddings',
            json={"model": "llama3", "prompt": text}
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding", None)
        if embedding is None or not isinstance(embedding, list):
            return None
        return list(map(float, embedding))  # ensure it's a list of floats
    except Exception as e:
        #print(f"Embedding failed for text: {text[:60]}... \nError: {e}")
        return None

df_flat["embedding"] = df_flat["answer"].map(get_llama3_embedding)
df_flat = df_flat[df_flat["embedding"].notnull()]


# Convert to 2D array for model training
X = np.vstack(df_flat["embedding"].values)
y = df_flat["answer_source"].values