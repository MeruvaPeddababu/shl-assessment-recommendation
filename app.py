import pandas as pd
import numpy as np
import gradio as gr
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load dataset with embeddings
df = pd.read_csv("shl_assessments_with_embeddings.csv")
df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

# Load transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Helper: extract minutes
def extract_duration(duration_str):
    match = re.search(r"(\d+)\s*min", str(duration_str))
    return int(match.group(1)) if match else float("inf")

# Core function
def recommend_with_json(query, duration_limit):
    query_embedding = model.encode(query)
    embeddings = np.stack(df["embedding"].values)
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    df_copy = df.copy()
    df_copy["similarity"] = similarities
    df_copy["duration_num"] = df_copy["duration"].apply(extract_duration)
    filtered = df_copy[df_copy["duration_num"] <= duration_limit]
    top_results = filtered.sort_values("similarity", ascending=False).head(10)

    # Table format (for display)
    table_df = top_results[[
        "name", "url", "remote_testing", "adaptive", "duration", "test_type"
    ]].rename(columns={
        "name": "Assessment Name",
        "url": "URL",
        "remote_testing": "Remote Testing",
        "adaptive": "Adaptive/IRT",
        "duration": "Duration",
        "test_type": "Test Type"
    })

    # JSON format (for copying or API-like output)
    json_result = {
        "query": query,
        "duration_limit": duration_limit,
        "results": []
    }

    for _, row in top_results.iterrows():
        json_result["results"].append({
            "assessment_name": row["name"],
            "url": row["url"],
            "remote_testing": row["remote_testing"],
            "adaptive": row["adaptive"],
            "duration": row["duration"],
            "test_type": row["test_type"],
            "similarity_score": round(row["similarity"], 3)
        })

    # Pretty JSON string
    json_output = json.dumps(json_result, indent=2)
    return table_df, json_output

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🔍 SHL Assessment Recommender")
    gr.Markdown("Enter a job description or query and get smart SHL test suggestions. You get a table and JSON output!")

    with gr.Row():
        query_input = gr.Textbox(label="Job Description / Query", placeholder="e.g. Hiring for analysts, need cognitive and personality tests", lines=3)
        duration_input = gr.Slider(10, 90, value=45, step=5, label="Max Duration (minutes)")

    submit_btn = gr.Button("🔎 Recommend")

    with gr.Row():
        output_table = gr.Dataframe(label="Recommended SHL Assessments (Table View)")
    
    output_json = gr.Code(label="📦 JSON Output (API Style)", language="json")

    submit_btn.click(fn=recommend_with_json, inputs=[query_input, duration_input], outputs=[output_table, output_json])

demo.launch()
