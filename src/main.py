import joblib
import pandas as pd
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, File,  UploadFile
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "svd_lof.joblib"

class ScoreRequest(BaseModel):
    records: List[Dict[str, Any]]
    top_k: Optional[int] = 10

@asynccontextmanager
async def lifespan(app: FastAPI):
    import __main__
    from src.custom_transformers import DropColumns, log_eps  # noqa: F401
    setattr(__main__, "DropColumns", DropColumns)
    setattr(__main__, "log_eps", log_eps)
    bundle = joblib.load(MODEL_PATH)

    app.state.pipe = bundle
    print("[MODEL] Loaded Pipeline steps:", [name for name, _ in bundle.steps])

    yield

app = FastAPI(title="LOF Scoring API", version="0.1", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score_csv")
async def score_csv(file: UploadFile = File(...), top_k: int = 10):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # đảm bảo numeric đúng dtype
    for c in ["DMBTR", "WRBTR"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df['anomaly_scored'] = -app.state.pipe.score_samples(df)
    df_return = (
        df.sort_values("anomaly_scored", ascending=False)[["BELNR", "anomaly_scored"]]
        .reset_index(drop=True)
    )
    return df_return.to_dict(orient="records")

