import os
import re
from typing import Optional, List

import duckdb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

PARQUET_PATH = os.environ.get("PARQUET_PATH", "/data/data.parquet")
API_KEY = os.environ.get("API_KEY")
ALLOW_ORIGINS = os.environ.get("ALLOW_ORIGINS", "*")

app = FastAPI(title="Parquet API (DuckDB)", version="0.4.2")

# CORS
origins = [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helpers
def _ensure_parquet_exists():
    if not os.path.exists(PARQUET_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"Parquet introuvable: {PARQUET_PATH}. Upload via /upload ou définis PARQUET_PATH."
        )

def _duck():
    """Connexion DuckDB prête à lire des parquets locaux et distants."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("PRAGMA threads=" + str(os.cpu_count() or 4))
    return con

def df_records_safe(df: pd.DataFrame):
    """
    Nettoyage total avant JSON:
    - remplace ±Inf -> NaN
    - convertit NaN/pd.NA/NaT -> None
    - convertit les dtypes pandas/numpy vers types Python natifs
    """
    if df is None:
        return []
    # Remplacer inf par NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # Remplacer toutes valeurs nulles (NaN, pd.NA, NaT) par None
    df = df.where(pd.notnull(df), None)
    # S'assure que les colonnes flottantes deviennent des objets Python natifs
    for col in df.columns:
        # Forcer object pour éviter des scalaires numpy non sérialisables
        if pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(object)
    return jsonable_encoder(df.to_dict(orient="records"))

def _json(data):
    """Encodage JSON sûr pour dict/list/valeurs simples."""
    return JSONResponse(content=jsonable_encoder(data))

# Routes
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/schema")
def schema():
    _ensure_parquet_exists()
    con = _duck()
    try:
        df = con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [PARQUET_PATH]
        ).fetchdf()
        cols = [{"name": r["column_name"], "type": r["column_type"]} for _, r in df.iterrows()]
        return {"path": PARQUET_PATH, "columns": cols}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()

@app.get("/preview")
def preview(limit: int = Query(50, ge=1, le=5000)):
    _ensure_parquet_exists()
    con = _duck()
    try:
        df = con.execute(
            "SELECT * FROM read_parquet(?) LIMIT ?",
            [PARQUET_PATH, limit]
        ).fetchdf()
        return _json({"rows": df_records_safe(df), "limit": limit})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()

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

    con = _duck()
    try:
        if "read_parquet(" not in safe_sql.lower():
            con.execute("CREATE OR REPLACE VIEW parquet_table AS SELECT * FROM read_parquet(?)", [PARQUET_PATH])
            result = con.execute(safe_sql).fetchdf()
        else:
            result = con.execute(safe_sql, [PARQUET_PATH]).fetchdf()

        return _json({"rows": df_records_safe(result)})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        con.close()

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

# ---------------------------------------------------------------------------
#             >> Endpoints API lisibles au navigateur <<
# ---------------------------------------------------------------------------

@app.get("/count")
def count_rows(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="X-API-Key manquante ou invalide")
    _ensure_parquet_exists()
    con = _duck()
    try:
        n = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [PARQUET_PATH]).fetchone()[0]
        return {"rows": int(n)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()

# --- config tri + recherche ---
_ALLOWED_ORDER_BY = {
    "uid", "acheteur_nom", "montant", "dateNotification",
    "acheteur_region_nom", "titulaire_nom", "procedure"
}

_TEXT_COLS = [
    "objet",          # description du marché
    "titulaire_nom",  # entreprise attributaire
    "acheteur_nom",   # acheteur
    "codeCPV",        # code CPV
    "procedure"       # type de procédure
]

def _escape_like_token(t: str) -> str:
    # pour ILIKE ... ESCAPE '\'
    return t.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")

@app.get("/rows")
def rows(
    x_api_key: Optional[str] = Header(None),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),

    q: Optional[str] = Query(None, description="mots-clés (AND) sur colonnes texte"),
    awardee: Optional[List[str]] = Query(None, description="multi: ?awardee=A&awardee=B"),

    acheteur_region_nom: Optional[str] = None,
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
    min_montant: Optional[float] = None,
    max_montant: Optional[float] = None,
    order_by: str = Query("dateNotification"),
    order_dir: str = Query("DESC"),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="X-API-Key manquante ou invalide")
    _ensure_parquet_exists()

    order_by = order_by if order_by in _ALLOWED_ORDER_BY else "dateNotification"
    order_dir = "ASC" if str(order_dir).upper() == "ASC" else "DESC"

    where = []
    params: List[object] = []

    # filtres classiques
    if acheteur_region_nom:
        where.append("acheteur_region_nom = ?")
        params.append(acheteur_region_nom)
    if date_from:
        where.append("dateNotification >= DATE ?")
        params.append(date_from)
    if date_to:
        where.append("dateNotification <= DATE ?")
        params.append(date_to)
    if min_montant is not None:
        where.append("montant >= ?")
        params.append(min_montant)
    if max_montant is not None:
        where.append("montant <= ?")
        params.append(max_montant)

    # q : AND entre tokens, OR entre colonnes texte
    if q:
        raw_tokens = [t for t in re.split(r"\s+", q.strip()) if t]
        tokens = [t for t in raw_tokens if len(t) >= 3][:4]  # garde-fous perfs
        for t in tokens:
            safe_like = f"%{_escape_like_token(t)}%"
            ors = []
            for col in _TEXT_COLS:
                ors.append(f"{col} ILIKE ? ESCAPE '\\'")
                params.append(safe_like)
            where.append("(" + " OR ".join(ors) + ")")

    # awardee multi
    if awardee:
        vals = [a for a in awardee if a and a.strip()]
        if vals:
            where.append("titulaire_nom IN (" + ",".join(["?"] * len(vals)) + ")")
            params.extend(vals)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    con = _duck()
    try:
        # --- un seul scan parquet + total en fenêtre ---
        page_sql = f"""
            WITH base AS (
                SELECT
                    uid,
                    acheteur_nom,
                    CASE WHEN isfinite(montant) THEN montant ELSE NULL END AS montant,
                    dateNotification,
                    acheteur_region_nom,
                    titulaire_nom,
                    procedure,
                    objet,
                    CASE WHEN isfinite(acheteur_latitude)  THEN acheteur_latitude  ELSE NULL END AS acheteur_latitude,
                    CASE WHEN isfinite(acheteur_longitude) THEN acheteur_longitude ELSE NULL END AS acheteur_longitude
                FROM read_parquet(?)
                {where_sql}
            )
            SELECT
                base.*,
                COUNT(*) OVER() AS __total
            FROM base
            ORDER BY {order_by} {order_dir}
            LIMIT ? OFFSET ?
        """
        df = con.execute(page_sql, [PARQUET_PATH, *params, limit, offset]).fetchdf()

        # total robuste (avec fallback si __total absent)
        if len(df) and "__total" in df.columns:
            total = int(df["__total"].max())
            df = df.drop(columns=["__total"])
        else:
            # Fallback (rare) : compte séparé
            total_sql = f"SELECT COUNT(*) FROM read_parquet(?) {where_sql}"
            total = int(con.execute(total_sql, [PARQUET_PATH, *params]).fetchone()[0])

        return _json({
            "total": total,
            "limit": int(limit),
            "offset": int(offset),
            "rows": df_records_safe(df),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()
