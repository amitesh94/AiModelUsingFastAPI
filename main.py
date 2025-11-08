import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis API")

# Define request model
class TextIn(BaseModel):
    text: str


class BatchTextIn(BaseModel):
    texts: list[str]


class ThresholdIn(BaseModel):
    text: str
    threshold: float = 0.5

# Initialize model components
pipeline = None

@app.on_event("startup")
async def load_model():
    """Load the model during startup."""
    global pipeline
    
    try:
        logger.info("Importing transformers library...")
        from transformers import pipeline
        
        logger.info("Loading sentiment analysis model... (this may take a moment)")
        pipeline = pipeline("sentiment-analysis")
        logger.info("✅ Model loaded successfully!")
        
    except ImportError as e:
        logger.error(f"Failed to import transformers: {e}")
        logger.error("Try running: pip install transformers torch")
        return
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

@app.get("/health")
async def health_check():
    """Check if the model is loaded and ready."""
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs for details."
        )
    return {"status": "healthy", "model": "ready"}

@app.post("/predict")
async def predict(input: TextIn):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Sentiment model is not available. Check server logs.")
    
    result = pipeline(input.text)
    return {"label": result[0]['label'], "score": result[0]['score']}


@app.get("/models")
async def models_status():
    """Return status about loaded models and library versions."""
    loaded = pipeline is not None
    info = {
        "loaded": loaded,
    }
    try:
        import transformers
        info["transformers"] = getattr(transformers, "__version__", "unknown")
    except Exception:
        info["transformers"] = "not-installed"

    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:
        info["torch"] = "not-installed"

    return info


@app.post("/batch_predict")
async def batch_predict(batch: BatchTextIn):
    """Predict a batch of texts; returns list of {label, score} objects."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Sentiment model is not available. Check server logs.")

    try:
        results = pipeline(batch.texts)
        # results is a list of prediction dicts
        return [{"label": r.get("label"), "score": r.get("score")} for r in results]
    except Exception as e:
        logger.exception("Batch prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_threshold")
async def predict_with_threshold(req: ThresholdIn):
    """Predict and only return label if score >= threshold; otherwise label as 'neutral'."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Sentiment model is not available. Check server logs.")

    try:
        result = pipeline(req.text)[0]
        score = float(result.get("score", 0))
        label = result.get("label") if score >= req.threshold else "neutral"
        return {"label": label, "score": round(score, 3), "threshold": req.threshold}
    except Exception as e:
        logger.exception("Threshold prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/version")
async def version_info():
    """Return versions of key libraries."""
    data = {}
    try:
        import transformers
        data["transformers"] = getattr(transformers, "__version__", "unknown")
    except Exception:
        data["transformers"] = "not-installed"
    try:
        import torch
        data["torch"] = getattr(torch, "__version__", "unknown")
    except Exception:
        data["torch"] = "not-installed"
    return data
