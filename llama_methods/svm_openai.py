from datasets import load_dataset
import pandas as pd
import numpy as np
import openai
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

df_flat = df_flat.sample(1000, random_state=42)

def get_openai_embedding(text, model="text-embedding-3-small"):
    try:
        response = openai.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Failed to embed: {text[:60]}...\nError: {e}")
        return None

tqdm.pandas()
df_flat["embedding"] = df_flat["answer"].progress_apply(get_openai_embedding)

df_flat = df_flat[df_flat["embedding"].notnull()]

X = np.vstack(df_flat["embedding"].values)
y = df_flat["answer_source"].values

df_flat.to_pickle("hc3_with_embeddings.pkl")
