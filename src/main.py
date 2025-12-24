# src/main.py
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "svd_lof.joblib"

class ScoreRequest(BaseModel):
    records: List[Dict[str, Any]]
    top_k: Optional[int] = 10

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    import __main__
    from src.custom_transformers import DropColumns, log_eps  # noqa: F401

    # patch để unpickle bundle cũ đang trỏ __main__.DropColumns / __main__.log_eps
    setattr(__main__, "DropColumns", DropColumns)
    setattr(__main__, "log_eps", log_eps)

    bundle = joblib.load(MODEL_PATH)
    app.state.feature_pipe = bundle["feature_pipe"]
    app.state.lof = bundle["lof"]

    yield

    # --- shutdown (optional) ---
    # del app.state.feature_pipe
    # del app.state.lof

app = FastAPI(title="SVD+LOF Scoring API", version="0.1", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(req: ScoreRequest):
    if not req.records:
        raise HTTPException(status_code=400, detail="records is empty")

    df = pd.DataFrame(req.records)

    try:
        Z = app.state.feature_pipe.transform(df)
        scores = -app.state.lof.score_samples(Z)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transform/score failed: {e}")

    s = pd.Series(scores)
    idx = s.sort_values(ascending=False).index.tolist()
    k = min(req.top_k or 10, len(idx))
    top_idx = idx[:k]

    return {
        "n": len(scores),
        "top_k": k,
        "top_index": top_idx,
        "top_scores": [float(scores[i]) for i in top_idx],
    }
# thêm vào src/main.py
from fastapi import UploadFile, File
import io

@app.post("/score_csv")
async def score_csv(file: UploadFile = File(...), top_k: int = 10):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # đảm bảo numeric đúng dtype
    for c in ["DMBTR", "WRBTR"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    Z = app.state.feature_pipe.transform(df)
    scores = -app.state.lof.score_samples(Z)

    df['anomaly_scored'] = scores
    df_return = (
        df.sort_values("anomaly_scored", ascending=False)[["BELNR", "anomaly_scored"]]
        .reset_index(drop=True)
    )
    return df_return.to_dict(orient="records")

