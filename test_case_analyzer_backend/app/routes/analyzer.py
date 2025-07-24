import os
from flask import Blueprint, request, jsonify, send_file
from flask.views import MethodView
from flask_smorest import abort
from werkzeug.utils import secure_filename
from datetime import datetime
import tempfile
import pandas as pd
import requests
import sqlite3
import numpy as np
import tensorflow_hub as hub

# PUBLIC_INTERFACE
class USEEncoder:
    """Wrapper around Tensorflow Hub USE with in-memory singleton initialization."""
    _model = None

    @classmethod
    def load_model(cls):
        if cls._model is None:
            cls._model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        return cls._model

    @classmethod
    def encode(cls, texts):
        model = cls.load_model()
        return np.array(model(texts))

# Database helper
def get_db_connection():
    # Read SQLite location from env
    db_path = os.environ.get("SQLITE_DB")
    if not db_path:
        raise RuntimeError("SQLITE_DB env variable not found!")
    return sqlite3.connect(db_path)

# Helper functions
def save_uploaded_file(file_storage, upload_folder):
    filename = secure_filename(file_storage.filename)
    save_path = os.path.join(upload_folder, f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}")
    file_storage.save(save_path)
    return save_path, filename

def parse_excel_test_cases(file_path):
    """Parse Excel and extract test case names and (optionally) steps."""
    df = pd.read_excel(file_path)
    # Heuristic: find likely columns for ID/name/desc
    cols = [c for c in df.columns if "name" in c.lower() or "desc" in c.lower() or "case" in c.lower()]
    if not cols:
        raise ValueError("Cannot detect test case name/description column.")
    # Use only the most likely column for text comparison
    use_col = cols[0]
    return df[use_col].astype(str).tolist(), use_col

def insert_file_and_cases(file_path, filename, test_cases):
    conn = get_db_connection()
    cur = conn.cursor()
    # Insert file record
    cur.execute("INSERT INTO test_case_files (filename, upload_date) VALUES (?, ?)", (filename, datetime.utcnow()))
    file_id = cur.lastrowid
    # Insert all test cases
    for t in test_cases:
        cur.execute("INSERT INTO test_cases (name, file_id) VALUES (?, ?)", (t, file_id))
    conn.commit()
    conn.close()
    return file_id

def fetch_test_cases_for_file(file_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT tc_id, name FROM test_cases WHERE file_id=?", (file_id,))
    res = cur.fetchall()
    conn.close()
    return [{"tc_id": r[0], "name": r[1]} for r in res]

def call_azure_openai_similarity(texts1, texts2):
    """PUBLIC_INTERFACE
    Calls Azure OpenAI LLM's embedding API for semantic similarity.
    Returns similarity matrix np.ndarray.
    """
    AZURE_API_KEY = os.environ.get("AZURE_OPENAI_KEY")
    AZURE_API_BASE = os.environ.get("AZURE_OPENAI_ENDPOINT")
    DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "text-embedding-ada-002")
    if not all([AZURE_API_KEY, AZURE_API_BASE, DEPLOYMENT]):
        raise RuntimeError("Missing Azure OpenAI configuration in .env")
    # Build API URL
    api_url = f"{AZURE_API_BASE}/openai/deployments/{DEPLOYMENT}/embeddings?api-version=2023-05-15"
    # POST texts1+2 in two calls, then compute sim matrix
    def get_embeds(batch):
        response = requests.post(
            api_url,
            headers={"api-key": AZURE_API_KEY, "Content-Type": "application/json"},
            json={"input": batch}
        )
        resp = response.json()
        if 'data' not in resp:
            raise RuntimeError(f"Azure response missing embeds: {resp}")
        return np.array([o["embedding"] for o in resp["data"]])
    embeds1 = get_embeds(texts1)
    embeds2 = get_embeds(texts2)
    # Cosine similarity
    norm1 = np.linalg.norm(embeds1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeds2, axis=1, keepdims=True)
    dot = np.dot(embeds1, embeds2.T)
    sim = dot / (norm1 * norm2.T + 1e-10)
    return sim

# Flask Blueprint
blp = Blueprint("Analyzer", __name__, url_prefix="/analyze", description="Test Case Analysis endpoints")

@blp.route("/upload", methods=["POST"])
class UploadView(MethodView):
    """PUBLIC_INTERFACE
    Handles uploading two Excel files.
    """
    def post(self):
        if "file1" not in request.files or "file2" not in request.files:
            abort(400, message="Missing file1 or file2")
        # Save uploads
        upload_folder = tempfile.mkdtemp()
        path1, fname1 = save_uploaded_file(request.files["file1"], upload_folder)
        path2, fname2 = save_uploaded_file(request.files["file2"], upload_folder)
        # Parse cases
        try:
            tcs1, col1 = parse_excel_test_cases(path1)
            tcs2, col2 = parse_excel_test_cases(path2)
        except Exception as e:
            abort(400, message=f"Excel parse error: {e}")

        # Insert in DB
        file1_id = insert_file_and_cases(path1, fname1, tcs1)
        file2_id = insert_file_and_cases(path2, fname2, tcs2)
        return jsonify({"file1_id": file1_id, "file2_id": file2_id, "file1_column": col1, "file2_column": col2})

@blp.route("/compare", methods=["POST"])
class CompareView(MethodView):
    """PUBLIC_INTERFACE
    Compares two uploaded test case files using selected method (USE or Azure OpenAI).
    """
    def post(self):
        data = request.get_json()
        file1_id = data.get("file1_id")
        file2_id = data.get("file2_id")
        method = data.get("method", "use")  # "use" or "azure"
        if not file1_id or not file2_id:
            abort(400, message="file1_id and file2_id are required")
        # Load cases
        tcs1 = fetch_test_cases_for_file(file1_id)
        tcs2 = fetch_test_cases_for_file(file2_id)
        tc_names1 = [tc["name"] for tc in tcs1]
        tc_names2 = [tc["name"] for tc in tcs2]

        try:
            if method == "azure":
                sim_matrix = call_azure_openai_similarity(tc_names1, tc_names2)
            else:
                enc1 = USEEncoder.encode(tc_names1)
                enc2 = USEEncoder.encode(tc_names2)
                dot = np.dot(enc1, enc2.T)
                norm1 = np.linalg.norm(enc1, axis=1, keepdims=True)
                norm2 = np.linalg.norm(enc2, axis=1, keepdims=True)
                sim_matrix = dot / (norm1 * norm2.T + 1e-10)
        except Exception as e:
            abort(500, message=f"Semantic embedding error: {e}")

        # Find duplicates/similar/unique
        threshold_duplicate = 0.97
        threshold_similar = 0.80
        results = []
        for i, t1 in enumerate(tc_names1):
            best_j = np.argmax(sim_matrix[i])
            sim_score = sim_matrix[i, best_j]
            if sim_score >= threshold_duplicate:
                cat = "duplicate"
            elif sim_score >= threshold_similar:
                cat = "similar"
            else:
                cat = "unique"
            results.append({
                "file1_tc": t1,
                "file2_tc": tc_names2[best_j],
                "similarity": float(sim_score),
                "category": cat
            })
        # Cases in file2 not matched to any in file1 as unique to file2
        matched_j = {np.argmax(sim_matrix[i]) for i in range(sim_matrix.shape[0])}
        unique2 = [j for j in range(len(tc_names2)) if j not in matched_j]
        uniques2_list = [{"file2_tc": tc_names2[j], "category": "unique"} for j in unique2]

        # Store results (could be extended to db, here as in-memory for return)
        return jsonify({
            "analysis": results,
            "unique2": uniques2_list,
            "method": method
        })

@blp.route("/download", methods=["GET"])
class DownloadView(MethodView):
    """PUBLIC_INTERFACE
    Download analysis result for given file ids and method as Excel file."""
    def get(self):
        file1_id = request.args.get("file1_id")
        file2_id = request.args.get("file2_id")
        method = request.args.get("method", "use")
        if not file1_id or not file2_id:
            abort(400, message="Provide file1_id, file2_id as query params")
        # Repeat comparison for deterministic output
        tcs1 = fetch_test_cases_for_file(file1_id)
        tcs2 = fetch_test_cases_for_file(file2_id)
        tc_names1 = [tc["name"] for tc in tcs1]
        tc_names2 = [tc["name"] for tc in tcs2]
        try:
            if method == "azure":
                sim_matrix = call_azure_openai_similarity(tc_names1, tc_names2)
            else:
                enc1 = USEEncoder.encode(tc_names1)
                enc2 = USEEncoder.encode(tc_names2)
                dot = np.dot(enc1, enc2.T)
                norm1 = np.linalg.norm(enc1, axis=1, keepdims=True)
                norm2 = np.linalg.norm(enc2, axis=1, keepdims=True)
                sim_matrix = dot / (norm1 * norm2.T + 1e-10)
        except Exception as e:
            abort(500, message=f"Semantic embedding error: {e}")

        threshold_duplicate = 0.97
        threshold_similar = 0.80
        results = []
        for i, t1 in enumerate(tc_names1):
            best_j = np.argmax(sim_matrix[i])
            sim_score = sim_matrix[i, best_j]
            if sim_score >= threshold_duplicate:
                cat = "duplicate"
            elif sim_score >= threshold_similar:
                cat = "similar"
            else:
                cat = "unique"
            results.append({
                "file1_tc": t1,
                "file2_tc": tc_names2[best_j],
                "similarity": float(sim_score),
                "category": cat
            })
        matched_j = {np.argmax(sim_matrix[i]) for i in range(sim_matrix.shape[0])}
        unique2 = [j for j in range(len(tc_names2)) if j not in matched_j]
        uniques2_list = [{"file2_tc": tc_names2[j], "category": "unique"} for j in unique2]

        # Compose to Excel
        df1 = pd.DataFrame(results)
        df2 = pd.DataFrame(uniques2_list)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmpf:
            with pd.ExcelWriter(tmpf.name) as writer:
                df1.to_excel(writer, index=False, sheet_name="MatchedCases")
                df2.to_excel(writer, index=False, sheet_name="UniqueToFile2")
            tmpf.seek(0)
            tmpfile_path = tmpf.name
        return send_file(tmpfile_path, as_attachment=True, download_name="semantic_analysis.xlsx")

