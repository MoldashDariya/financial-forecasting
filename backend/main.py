# FastAPI backend of the dashboard.
# What it does: login/register (sessions kept in SQLite), CSV upload -> ML pipeline (services/ml_pipeline.py),
# saving every user's uploads + predictions, and serving the bundled diploma results to the frontend.
# Run it from the project root:  uvicorn backend.main:app --reload --port 8012
# (the Vite dev server proxies /api to port 8012, see vite.config.js)
from datetime import datetime, timezone
from typing import Optional
import hashlib
import hmac
import sqlite3
import secrets

from fastapi import Depends, FastAPI, Header, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import shutil
from pathlib import Path
import math

import pandas as pd

try:
    # works both when started from /backend and from the project root
    from ml_pipeline.run_pipeline import run_pipeline, load_company_csv, pipeline_dependency_status
except ModuleNotFoundError:
    from backend.ml_pipeline.run_pipeline import run_pipeline, load_company_csv, pipeline_dependency_status

app = FastAPI(title="Financial Forecasting Backend")

# Only the local Vite dev servers may call the API (Vite jumps to 5174/5175 if 5173 is busy)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://localhost:5174", "http://localhost:5175",
        "http://127.0.0.1:5173", "http://127.0.0.1:5174", "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths. The bundled result JSONs live in the frontend folder (src/) because the dashboard imports them directly,
# the backend just serves the same files
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_SRC_DIR = BASE_DIR.parent / "src"
BUNDLED_DASHBOARD_PATH = FRONTEND_SRC_DIR / "diploma_dashboard_data.json"
BUNDLED_FORECAST_PAGE_PATH = FRONTEND_SRC_DIR / "forecast_page_data.json"
UPLOAD_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "data" / "results"
DB_PATH = BASE_DIR / "data" / "app.db"
# create the data folders on a fresh clone (raw/ and processed/ are not stored in git)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class UserContext(BaseModel):
    id: int
    username: str
    role: str


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# Passwords are never stored as plain text: PBKDF2-SHA256, 120k iterations, random salt.
# Saved as "salt$hash"
def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return f"{salt}${digest.hex()}"


def verify_password(password, password_hash):
    try:
        salt, expected = password_hash.split("$", 1)
    except ValueError:
        return False
    actual = hash_password(password, salt).split("$", 1)[1]
    # constant-time comparison, so response time can't leak how much of the hash matched
    return hmac.compare_digest(actual, expected)


# Creates the tables on startup if they don't exist yet:
# users, sessions, company_uploads (one row per uploaded file), company_rows (its cleaned rows),
# predictions (the model results of an upload as JSON)
def init_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS company_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                company_name TEXT NOT NULL,
                clean_file TEXT,
                source TEXT,
                pipeline_mode TEXT,
                rows_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS company_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id INTEGER NOT NULL,
                row_index INTEGER NOT NULL,
                date TEXT,
                revenue REAL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(upload_id) REFERENCES company_uploads(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id INTEGER NOT NULL,
                model_results_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(upload_id) REFERENCES company_uploads(id)
            )
            """
        )

        # Two demo accounts for testing. These passwords are only defaults for local use,
        # set DIPLOMA_ADMIN_PASSWORD / DIPLOMA_USER_PASSWORD before putting this on a public server
        seed_users = [
            ("admin", os.getenv("DIPLOMA_ADMIN_PASSWORD", "admin123"), "admin"),
            ("user", os.getenv("DIPLOMA_USER_PASSWORD", "user123"), "user"),
        ]
        for username, password, role in seed_users:
            exists = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if not exists:
                conn.execute(
                    "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                    (username, hash_password(password), role, utc_now_iso()),
                )


@app.on_event("startup")
def startup_event():
    init_database()


def user_payload(row):
    return {"id": int(row["id"]), "username": row["username"], "role": row["role"]}


# Auth dependency for protected endpoints: reads the Bearer token, looks it up in `sessions`, returns the user.
# Sessions don't expire at the moment (todo: check created_at if this ever goes to production)
def get_current_user(
    authorization: Optional[str] = Header(default=None),
    conn: sqlite3.Connection = Depends(get_db),
):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Login required.")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Login required.")

    row = conn.execute(
        """
        SELECT users.id, users.username, users.role
        FROM sessions
        JOIN users ON users.id = sessions.user_id
        WHERE sessions.token = ?
        """,
        (token,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")

    return UserContext(**user_payload(row))


# Same, but only admins get through (403 for everybody else)
def get_current_admin(current_user: UserContext = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user


# Stores one upload: a row in company_uploads, all cleaned rows in company_rows and the model results in predictions.
# That is why an upload is still there after logging out and in again
def save_company_upload_to_db(conn, current_user, payload, clean_df, filename):
    company_name = payload.get("company") or Path(filename).stem
    source = payload.get("source")
    pipeline_mode = payload.get("pipeline_mode")
    clean_file = payload.get("uploaded_clean_file") or payload.get("clean_file")
    created_at = utc_now_iso()
    clean_rows = _clean_rows_for_payload(clean_df).to_dict(orient="records")

    cursor = conn.execute(
        """
        INSERT INTO company_uploads
            (user_id, filename, company_name, clean_file, source, pipeline_mode, rows_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            current_user.id,
            filename,
            str(company_name),
            str(clean_file) if clean_file else None,
            source,
            pipeline_mode,
            int(len(clean_rows)),
            created_at,
        ),
    )
    upload_id = cursor.lastrowid

    for index, row in enumerate(clean_rows):
        conn.execute(
            """
            INSERT INTO company_rows (upload_id, row_index, date, revenue, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                upload_id,
                index,
                row.get("date"),
                row.get("revenue"),
                json.dumps(sanitize_for_json(row), ensure_ascii=False),
            ),
        )

    prediction_payload = payload.get("model_results") or {
        key: value
        for key, value in payload.items()
        if key not in {"preview", "clean_rows"}
    }
    conn.execute(
        """
        INSERT INTO predictions (upload_id, model_results_json, created_at)
        VALUES (?, ?, ?)
        """,
        (
            upload_id,
            json.dumps(sanitize_for_json(prediction_payload), ensure_ascii=False),
            created_at,
        ),
    )
    conn.commit()
    return upload_id


# The opposite direction: rebuilds the same JSON the upload endpoint returned from what is in the DB,
# so the frontend can treat old and new uploads exactly the same way
def build_upload_payload_from_db(conn, upload):
    rows = conn.execute(
        """
        SELECT row_index, payload_json
        FROM company_rows
        WHERE upload_id = ?
        ORDER BY row_index
        """,
        (int(upload["id"]),),
    ).fetchall()
    clean_rows = [json.loads(row["payload_json"]) for row in rows]
    preview = clean_rows[:10]
    available_columns = list(clean_rows[0].keys()) if clean_rows else []

    prediction = conn.execute(
        """
        SELECT model_results_json
        FROM predictions
        WHERE upload_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (int(upload["id"]),),
    ).fetchone()
    prediction_payload = json.loads(prediction["model_results_json"]) if prediction else {}

    if isinstance(prediction_payload, dict) and prediction_payload.get("status") and prediction_payload.get("source"):
        payload = dict(prediction_payload)
    else:
        payload = {"model_results": prediction_payload}

    payload.update(
        {
            "status": "success",
            "upload_id": int(upload["id"]),
            "filename": upload["filename"],
            "company": upload["company_name"],
            "clean_file": upload["clean_file"],
            "source": payload.get("source") or upload["source"] or "runtime_pipeline",
            "pipeline_mode": payload.get("pipeline_mode") or upload["pipeline_mode"],
            "rows": int(upload["rows_count"]),
            "preview": preview,
            "clean_rows": clean_rows,
            "available_columns": available_columns,
            "created_at": upload["created_at"],
            "owner": upload["username"] if "username" in upload.keys() else None,
        }
    )
    return sanitize_for_json(payload)


# NaN and inf are not valid JSON, so they become None (null). Works recursively on dicts and lists
def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


# Company name matching. A file is called something like 'Apple_final.csv' but the dashboard knows the company as 'Apple',
# so names are lowercased and stripped of filler words (final, corp, inc, ...) before they are compared
def normalize_company_key(value):
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def normalize_company_alias_key(value):
    text = Path(str(value)).stem.lower()
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    ignored_tokens = {
        "clean",
        "co",
        "company",
        "copy",
        "corp",
        "corporation",
        "csv",
        "data",
        "dataset",
        "final",
        "financial",
        "inc",
        "limited",
        "llc",
        "plc",
        "raw",
        "revenue",
        "quarterly",
    }
    tokens = [token for token in text.split() if token not in ignored_tokens]
    return "".join(tokens)


def _build_display_name_lookup(bundle_companies):
    lookup = {}
    # manual aliases for names that are too different to match automatically
    aliases = {
        "alphabet": "Google",
        "boa": "Bank of America",
        "bofa": "Bank of America",
        "bankamerica": "Bank of America",
        "blackberrylimited": "BlackBerry",
        "chevroncorp": "Chevron",
        "exxonmobil": "Exxon",
        "exxonmobilcorp": "Exxon",
        "gamestopcorp": "GameStop",
        "gapinc": "Gap",
        "hiltonworldwide": "Hilton",
        "jpmorgan": "JPMorgan",
        "jpmorganchase": "JPMorgan",
        "jpmorganchaseco": "JPMorgan",
        "marriottinternational": "Marriott",
        "microsoftcorp": "Microsoft",
        "nvidiacorp": "NVIDIA",
        "pelotoninteractive": "Peloton",
        "stitchfixinc": "Stitch Fix",
        "tripadvisorinc": "Tripadvisor",
        "visainc": "Visa",
        "zoomcommunications": "Zoom",
        "zoomvideocommunications": "Zoom",
    }

    for company in bundle_companies:
        lookup[normalize_company_key(company)] = company
        lookup[normalize_company_alias_key(company)] = company

    for alias, company in aliases.items():
        if company in bundle_companies:
            lookup[normalize_company_key(alias)] = company
            lookup[normalize_company_alias_key(alias)] = company

    return lookup


def _resolve_bundled_display_name(value, display_name_by_key, allow_partial=False):
    candidate_keys = [
        normalize_company_key(value),
        normalize_company_alias_key(value),
    ]
    for key in candidate_keys:
        if key in display_name_by_key:
            return display_name_by_key[key]

    if not allow_partial:
        return None

    for key in candidate_keys:
        if not key:
            continue
        matches = [
            (company_key, company)
            for company_key, company in display_name_by_key.items()
            if company_key and (key.startswith(company_key) or key.endswith(company_key))
        ]
        if matches:
            return sorted(matches, key=lambda item: len(item[0]), reverse=True)[0][1]

    return None


def _clean_rows_for_payload(df):
    clean_rows = df.copy()
    if "date" in clean_rows.columns:
        clean_rows["date"] = clean_rows["date"].astype(str)
    return clean_rows


# Two files count as the same data if their (date, revenue) columns are equal after rounding to 6 decimals
def _dataframes_match(left, right):
    required_cols = {"date", "revenue"}
    if not required_cols.issubset(left.columns) or not required_cols.issubset(right.columns):
        return False

    left_cmp = left.loc[:, ["date", "revenue"]].copy()
    right_cmp = right.loc[:, ["date", "revenue"]].copy()
    left_cmp["date"] = pd.to_datetime(left_cmp["date"], errors="coerce").astype(str)
    right_cmp["date"] = pd.to_datetime(right_cmp["date"], errors="coerce").astype(str)
    left_cmp["revenue"] = pd.to_numeric(left_cmp["revenue"], errors="coerce").round(6)
    right_cmp["revenue"] = pd.to_numeric(right_cmp["revenue"], errors="coerce").round(6)
    left_cmp = left_cmp.dropna().reset_index(drop=True)
    right_cmp = right_cmp.dropna().reset_index(drop=True)
    return left_cmp.equals(right_cmp)


# Shortcut for the demo: if the uploaded file is one of the companies that are already in the diploma results
# (same name or same data) we skip training and answer with the precomputed results.
# Returns None when it's a company we have never seen
def _load_precomputed_company_payload(uploaded_df, uploaded_name, exclude_path=None):
    processed_candidates = sorted(PROCESSED_DIR.glob("*_clean.csv"))
    name_key = normalize_company_key(uploaded_name)
    alias_name_key = normalize_company_alias_key(uploaded_name)
    bundled_dashboard = json.loads(BUNDLED_DASHBOARD_PATH.read_text()) if BUNDLED_DASHBOARD_PATH.exists() else {}
    bundle_companies = bundled_dashboard.get("meta", {}).get("companies", [])
    display_name_by_key = _build_display_name_lookup(bundle_companies)

    matched_processed_path = None
    match_reason = None
    for processed_path in processed_candidates:
        if exclude_path and processed_path.resolve() == Path(exclude_path).resolve():
            continue
        processed_df = pd.read_csv(processed_path)
        processed_slug = processed_path.stem.replace("_clean", "")
        processed_key = normalize_company_key(processed_slug)
        processed_alias_key = normalize_company_alias_key(processed_slug)
        if name_key and name_key in {processed_key, processed_alias_key}:
            matched_processed_path = processed_path
            match_reason = "processed_filename"
            break
        if alias_name_key and alias_name_key in {processed_key, processed_alias_key}:
            matched_processed_path = processed_path
            match_reason = "processed_filename_alias"
            break
        if _dataframes_match(uploaded_df, processed_df):
            matched_processed_path = processed_path
            match_reason = "processed_data"
            break

    company_slug = None
    if matched_processed_path is not None:
        company_slug = matched_processed_path.stem.replace("_clean", "")
        matched_display_name = _resolve_bundled_display_name(company_slug, display_name_by_key, allow_partial=True)
    else:
        matched_display_name = _resolve_bundled_display_name(uploaded_name, display_name_by_key)
        if matched_display_name:
            company_slug = normalize_company_alias_key(matched_display_name) or normalize_company_key(matched_display_name)
            match_reason = "bundled_company_filename"

    if not matched_display_name:
        return None

    clean_df = pd.read_csv(matched_processed_path) if matched_processed_path is not None else uploaded_df.copy()
    clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")
    clean_df = clean_df.dropna(subset=["date", "revenue"]).reset_index(drop=True)
    preview_cols = ["date", "revenue"] + [c for c in clean_df.columns if c not in ["date", "revenue"]][:5]
    preview = _clean_rows_for_payload(clean_df[preview_cols].head(10))
    clean_rows = _clean_rows_for_payload(clean_df)

    return sanitize_for_json(
        {
            "status": "success",
            "source": "bundled_dashboard_company",
            "pipeline_mode": "precomputed",
            "matched_company": company_slug,
            "bundled_company": matched_display_name,
            "company": company_slug,
            "filename": uploaded_name,
            "clean_file": str(matched_processed_path) if matched_processed_path is not None else None,
            "match_reason": match_reason,
            "rows": int(len(clean_df)),
            "preview": preview.to_dict(orient="records"),
            "clean_rows": clean_rows.to_dict(orient="records"),
            "available_columns": clean_rows.columns.tolist(),
        }
    )


# Second cache: the result of an earlier full pipeline run on the same data (data/results/*_full_metrics.json)
def _load_cached_runtime_pipeline_payload(clean_df, company_name, runtime_profile):
    # note: `safe` is not used in this function (the same line is repeated in _load_cached_metrics_payload where it is used)
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(company_name).strip().lower()).strip("_")[:60] or "uploaded_company"
    clean_path = PROCESSED_DIR / f"{company_name}_clean.csv"
    if not clean_path.exists():
        return None

    try:
        cached_clean_df = pd.read_csv(clean_path)
    except Exception:
        return None
    if not _dataframes_match(clean_df, cached_clean_df):
        return None

    return _load_cached_metrics_payload(company_name, runtime_profile)


def _load_cached_metrics_payload(company_name, runtime_profile):
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(company_name).strip().lower()).strip("_")[:60] or "uploaded_company"
    metric_paths = []
    exact_path = RESULTS_DIR / f"{safe}_full_metrics.json"
    if exact_path.exists():
        metric_paths.append(exact_path)
    metric_paths.extend(sorted(RESULTS_DIR.glob(f"{safe}_*_full_metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True))

    candidates = []
    for metrics_path in dict.fromkeys(metric_paths):
        try:
            cached_payload = json.loads(metrics_path.read_text())
        except Exception:
            continue
        cached_runtime_profile = cached_payload.get("runtime_profile")
        if cached_runtime_profile is not None and cached_runtime_profile != runtime_profile:
            continue
        variant = cached_payload.get("pipeline_environment", {}).get("variant")
        if variant not in {None, "notebook-parity backend pipeline"}:
            continue
        rows = cached_payload.get("all_model_metrics") or cached_payload.get("leaderboard") or []
        model_names = {str(row.get("model", "")) for row in rows}
        has_advanced_models = any(
            ("XGBoost" in model_name or "ARIMA" in model_name or "SARIMA" in model_name or "Prophet" in model_name)
            for model_name in model_names
        )
        # Pick the best cached file. Points for: same runtime profile (+4), made by the notebook-parity pipeline (+3),
        # exact file name (+1), has the advanced models like ARIMA / XGBoost / Prophet (+1)
        score = 0
        if cached_runtime_profile == runtime_profile:
            score += 4
        if variant == "notebook-parity backend pipeline":
            score += 3
        if metrics_path == exact_path:
            score += 1
        if has_advanced_models:
            score += 1
        candidates.append((score, metrics_path.stat().st_mtime, cached_payload))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


# ---- auth: login, register, me, logout ----
@app.post("/api/auth/login")
def login(credentials: LoginRequest, conn: sqlite3.Connection = Depends(get_db)):
    user = conn.execute(
        "SELECT id, username, password_hash, role FROM users WHERE username = ?",
        (credentials.username.strip(),),
    ).fetchone()
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")

    # random session token. The frontend keeps it in localStorage and sends it back as `Authorization: Bearer ...`
    token = secrets.token_urlsafe(32)
    conn.execute(
        "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, int(user["id"]), utc_now_iso()),
    )
    conn.commit()
    return {"token": token, "user": user_payload(user)}


@app.post("/api/auth/register")
def register(credentials: RegisterRequest, conn: sqlite3.Connection = Depends(get_db)):
    username = credentials.username.strip()
    password = credentials.password
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    exists = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if exists:
        raise HTTPException(status_code=409, detail="Username is already taken.")

    cursor = conn.execute(
        "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
        (username, hash_password(password), "user", utc_now_iso()),
    )
    token = secrets.token_urlsafe(32)
    conn.execute(
        "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, int(cursor.lastrowid), utc_now_iso()),
    )
    conn.commit()
    return {
        "token": token,
        "user": {"id": int(cursor.lastrowid), "username": username, "role": "user"},
    }


@app.get("/api/auth/me")
def get_me(current_user: UserContext = Depends(get_current_user)):
    return {"user": current_user.model_dump()}


@app.post("/api/auth/logout")
def logout(
    authorization: Optional[str] = Header(default=None),
    conn: sqlite3.Connection = Depends(get_db),
):
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
    return {"status": "success"}


# ---- uploads: a user sees only their own, an admin sees everyone's ----
@app.get("/api/uploads/my")
def my_uploads(
    current_user: UserContext = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
):
    uploads = conn.execute(
        """
        SELECT company_uploads.*, users.username
        FROM company_uploads
        JOIN users ON users.id = company_uploads.user_id
        WHERE company_uploads.user_id = ?
        ORDER BY company_uploads.created_at DESC, company_uploads.id DESC
        """,
        (current_user.id,),
    ).fetchall()
    return {
        "status": "success",
        "uploads": [build_upload_payload_from_db(conn, upload) for upload in uploads],
    }


@app.get("/api/uploads/all")
def all_uploads_for_admin(
    current_user: UserContext = Depends(get_current_admin),
    conn: sqlite3.Connection = Depends(get_db),
):
    uploads = conn.execute(
        """
        SELECT company_uploads.*, users.username
        FROM company_uploads
        JOIN users ON users.id = company_uploads.user_id
        ORDER BY company_uploads.created_at DESC, company_uploads.id DESC
        """
    ).fetchall()
    return {
        "status": "success",
        "uploads": [build_upload_payload_from_db(conn, upload) for upload in uploads],
    }


@app.delete("/api/uploads/{upload_id}")
def delete_my_upload(
    upload_id: int,
    current_user: UserContext = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
):
    upload = conn.execute(
        "SELECT id FROM company_uploads WHERE id = ? AND user_id = ?",
        (upload_id, current_user.id),
    ).fetchone()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload was not found for this user.")

    conn.execute("DELETE FROM predictions WHERE upload_id = ?", (upload_id,))
    conn.execute("DELETE FROM company_rows WHERE upload_id = ?", (upload_id,))
    conn.execute("DELETE FROM company_uploads WHERE id = ? AND user_id = ?", (upload_id, current_user.id))
    conn.commit()
    return {"status": "success", "deleted_upload_id": upload_id}


# ---- admin-only endpoints (the 'Admin DB' page) ----
@app.get("/api/admin/users")
def admin_users(
    current_user: UserContext = Depends(get_current_admin),
    conn: sqlite3.Connection = Depends(get_db),
):
    rows = conn.execute(
        """
        SELECT users.id, users.username, users.role, users.created_at,
               COUNT(company_uploads.id) AS uploads_count
        FROM users
        LEFT JOIN company_uploads ON company_uploads.user_id = users.id
        GROUP BY users.id
        ORDER BY users.id
        """
    ).fetchall()
    return {
        "status": "success",
        "users": [dict(row) for row in rows],
    }


@app.get("/api/admin/uploads")
def admin_uploads(
    current_user: UserContext = Depends(get_current_admin),
    conn: sqlite3.Connection = Depends(get_db),
):
    rows = conn.execute(
        """
        SELECT company_uploads.id, company_uploads.filename, company_uploads.company_name,
               company_uploads.clean_file, company_uploads.source, company_uploads.pipeline_mode,
               company_uploads.rows_count, company_uploads.created_at,
               users.username, users.role
        FROM company_uploads
        JOIN users ON users.id = company_uploads.user_id
        ORDER BY company_uploads.created_at DESC, company_uploads.id DESC
        """
    ).fetchall()
    return {
        "status": "success",
        "uploads": [dict(row) for row in rows],
    }


@app.get("/api/admin/uploads/{upload_id}")
def admin_upload_detail(
    upload_id: int,
    current_user: UserContext = Depends(get_current_admin),
    conn: sqlite3.Connection = Depends(get_db),
):
    upload = conn.execute(
        """
        SELECT company_uploads.*, users.username
        FROM company_uploads
        JOIN users ON users.id = company_uploads.user_id
        WHERE company_uploads.id = ?
        """,
        (upload_id,),
    ).fetchone()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload was not found.")

    rows = conn.execute(
        """
        SELECT row_index, date, revenue, payload_json
        FROM company_rows
        WHERE upload_id = ?
        ORDER BY row_index
        """,
        (upload_id,),
    ).fetchall()
    prediction = conn.execute(
        """
        SELECT model_results_json, created_at
        FROM predictions
        WHERE upload_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (upload_id,),
    ).fetchone()

    return sanitize_for_json(
        {
            "status": "success",
            "upload": dict(upload),
            "rows": [
                {
                    "row_index": int(row["row_index"]),
                    "date": row["date"],
                    "revenue": row["revenue"],
                    "payload": json.loads(row["payload_json"]),
                }
                for row in rows
            ],
            "prediction": {
                "created_at": prediction["created_at"],
                "model_results": json.loads(prediction["model_results_json"]),
            }
            if prediction
            else None,
        }
    )


# ---- misc: health check, pipeline status, default data ----
@app.get("/")
def home():
    return {"message": "Backend is working", "pipeline": "full ML + TS + macro/micro + 16-quarter forecast"}


@app.get("/api/pipeline/status")
def get_pipeline_status():
    return sanitize_for_json(pipeline_dependency_status())


# The default dashboard data: the JSON files with the results from the notebooks (21 companies).
# The forecast page data is merged into the same response
@app.get("/api/dashboard/default")
def get_default_dashboard():
    if not BUNDLED_DASHBOARD_PATH.exists() or not BUNDLED_FORECAST_PAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="Bundled dashboard JSON files were not found.")

    dashboard_payload = json.loads(BUNDLED_DASHBOARD_PATH.read_text())
    forecast_page_payload = json.loads(BUNDLED_FORECAST_PAGE_PATH.read_text())

    dashboard_payload["forecastPage"] = {
        "companies": forecast_page_payload.get("companies", []),
        "data": forecast_page_payload.get("data", {}),
        "scenarioLabels": forecast_page_payload.get("scenario_labels", {}),
    }

    return sanitize_for_json(
        {
            "status": "success",
            "source": "bundled_notebook_analysis",
            "dashboard": dashboard_payload,
        }
    )


# Main endpoint: upload a company CSV. What happens:
# 1) save the file  2) parse + clean it  3) is it a company from the diploma? -> reuse the precomputed results
# 4) was the same data run before? -> reuse the cached results  5) otherwise train everything for real (slow).
# fast_pipeline=true switches to the small 'balanced' grids without LSTM
@app.post("/api/upload")
def upload_csv(
    file: UploadFile = File(...),
    full_pipeline: bool = False, # not used anywhere (leftover), only fast_pipeline changes the behaviour
    fast_pipeline: bool = False,
    use_cache: bool = True,
    current_user: UserContext = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
):
    safe_filename = Path(file.filename).name
    # Path(...).name above removes any folders from the file name (no ../ tricks).
    # note: the file is saved in data/raw under its own name, so uploading 'nokia.csv' overwrites an existing nokia.csv
    file_path = UPLOAD_DIR / safe_filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    company_name = Path(safe_filename).stem

    # Save a clean normalized copy for debugging and frontend preview.
    try:
        clean_df = load_company_csv(file_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{safe_filename} is not a supported company revenue file. "
                "Upload a CSV with date and revenue columns, or a Refinitiv-style company financial export."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse {safe_filename}: {exc}") from exc

    clean_name = f"{company_name}_clean.csv"
    save_path = PROCESSED_DIR / clean_name
    # 'full' = big grids + LSTM (takes minutes), 'balanced' = small grids, no LSTM
    runtime_profile = "balanced" if fast_pipeline else "full"

    precomputed_payload = _load_precomputed_company_payload(clean_df, company_name, exclude_path=save_path)
    if precomputed_payload is not None:
        clean_df.to_csv(save_path, index=False)
        precomputed_payload["uploaded_clean_file"] = str(save_path)
        cached_model_results = (
            _load_cached_runtime_pipeline_payload(clean_df, company_name, runtime_profile)
            or _load_cached_runtime_pipeline_payload(
                clean_df,
                precomputed_payload.get("company", company_name),
                runtime_profile,
            )
            or _load_cached_metrics_payload(company_name, runtime_profile)
            or _load_cached_metrics_payload(precomputed_payload.get("company", company_name), runtime_profile)
        ) if use_cache else None
        if cached_model_results is not None:
            precomputed_payload["source"] = "runtime_pipeline"
            precomputed_payload["pipeline_mode"] = "cached_runtime_pipeline"
            precomputed_payload["model_results"] = cached_model_results
        precomputed_payload["upload_id"] = save_company_upload_to_db(
            conn,
            current_user,
            precomputed_payload,
            clean_df,
            safe_filename,
        )
        return precomputed_payload

    # not a diploma company -> maybe a cached run of the same data, otherwise train
    cached_model_results = _load_cached_runtime_pipeline_payload(clean_df, company_name, runtime_profile) if use_cache else None
    clean_df.to_csv(save_path, index=False)
    if cached_model_results is not None:
        model_results = cached_model_results
    else:
        try:
            # the slow part: tuning + fitting all models, then the recursive forecast until 2032 (see services/ml_pipeline.py)
            model_results = run_pipeline(str(file_path), company_name=company_name, runtime_profile=runtime_profile)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Pipeline could not process {safe_filename}: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Pipeline failed for {safe_filename}: {exc}") from exc

    preview_cols = ["date", "revenue"] + [c for c in clean_df.columns if c not in ["date", "revenue"]][:5]
    preview = clean_df[preview_cols].head(10).copy()
    preview["date"] = preview["date"].astype(str)
    clean_rows = clean_df.copy()
    clean_rows["date"] = clean_rows["date"].astype(str)

    response_payload = sanitize_for_json({
        "status": "success",
        "source": "runtime_pipeline",
        "pipeline_mode": "runtime_pipeline",
        "filename": safe_filename,
        "company": company_name,
        "clean_file": str(save_path),
        "rows": int(len(clean_df)),
        "preview": preview.to_dict(orient="records"),
        "clean_rows": clean_rows.to_dict(orient="records"),
        "available_columns": clean_rows.columns.tolist(),
        "model_results": model_results,
    })
    # save to SQLite so the upload shows up under 'My Data'
    response_payload["upload_id"] = save_company_upload_to_db(
        conn,
        current_user,
        response_payload,
        clean_df,
        safe_filename,
    )
    return response_payload
