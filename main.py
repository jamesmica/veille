import os
import re
from typing import Optional

import duckdb
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PARQUET_PATH = os.environ.get("PARQUET_PATH", "/data/data.parquet")
API_KEY = os.environ.get("API_KEY")
ALLOW_ORIGINS = os.environ.get("ALLOW_ORIGINS", "*")

app = FastAPI(title="Parquet API (DuckDB)", version="0.2.1")

origins = [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _ensure_parquet_exists():
    if not os.path.exists(PARQUET_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"Parquet introuvable: {PARQUET_PATH}. Upload via /upload ou définis PARQUET_PATH."
        )

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/schema")
def schema():
    _ensure_parquet_exists()
    try:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        df = con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [PARQUET_PATH]
        ).fetchdf()
        cols = [{"name": r["column_name"], "type": r["column_type"]} for _, r in df.iterrows()]
        return {"path": PARQUET_PATH, "columns": cols}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/preview")
def preview(limit: int = Query(50, ge=1, le=5000)):
    _ensure_parquet_exists()
    try:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        df = con.execute(
            "SELECT * FROM read_parquet(?) LIMIT ?",
            [PARQUET_PATH, limit]
        ).fetchdf()
        return JSONResponse(content={"rows": df.to_dict(orient="records"), "limit": limit})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE)

@app.get("/query")
def query(sql: str = Query(..., description="Requête SQL SELECT uniquement"),
          limit: Optional[int] = Query(None, ge=1, le=100000)):
    _ensure_parquet_exists()

    if not _SELECT_RE.match(sql) or ";" in sql:
        raise HTTPException(status_code=400, detail="Seulement une requête SELECT est autorisée.")

    safe_sql = sql.strip()
    if limit is not None and re.search(r"\blimit\b", safe_sql, flags=re.IGNORECASE) is None:
        safe_sql = f"{safe_sql} LIMIT {limit}"

    try:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("PRAGMA threads=" + str(os.cpu_count() or 4))

        if "read_parquet(" not in safe_sql.lower():
            # Vue pratique si l'utilisateur n'emploie pas read_parquet(...)
            con.execute("CREATE OR REPLACE VIEW parquet_table AS SELECT * FROM read_parquet(?)", [PARQUET_PATH])
            result = con.execute(safe_sql).fetchdf()
        else:
            # Si l'utilisateur a mis read_parquet(?), on lie le paramètre
            result = con.execute(safe_sql, [PARQUET_PATH]).fetchdf()

        return JSONResponse(content={"rows": result.to_dict(orient="records")})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload")
def upload_parquet(file: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="X-API-Key manquante ou invalide")
    if not file.filename.endswith(".parquet"):
        raise HTTPException(status_code=400, detail="Uploader un fichier .parquet")

    os.makedirs(os.path.dirname(PARQUET_PATH), exist_ok=True)
    try:
        contents = file.file.read()
        with open(PARQUET_PATH, "wb") as f:
            f.write(contents)
        return {"message": f"Upload OK → {PARQUET_PATH}", "size_bytes": len(contents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()
