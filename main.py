import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

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

class KeywordExtractionIn(BaseModel):
    text: str
    max_keywords: int = Field(default=5, gt=0, le=20)
    min_word_length: int = Field(default=3, gt=0)

class TextGenInput(BaseModel):
    prompt: str
    max_length: int = Field(default=100, gt=0, le=500)
    temperature: float = Field(default=0.7, gt=0, le=2.0)

class EntityExtractionIn(BaseModel):
    text: str
    include_types: List[str] = ["PER", "ORG", "LOC"]

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


@app.post("/extract-keywords")
async def extract_keywords(req: KeywordExtractionIn):
    """Extract key phrases and important words from text."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not available. Check server logs.")
    
    try:
        from transformers import pipeline as keyword_pipeline
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        import nltk
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Initialize keyword extractor
        keyword_extractor = keyword_pipeline(
            "feature-extraction",
            model="distilbert-base-uncased"
        )
        
        # Tokenize and clean text
        tokens = word_tokenize(req.text.lower())
        stop_words = set(stopwords.words('english'))
        words = [word for word in tokens 
                if word.isalnum() 
                and len(word) >= req.min_word_length 
                and word not in stop_words]
        
        # Get embeddings for words
        embeddings = keyword_extractor(words, padding=True, truncation=True)
        
        # Calculate word importance scores (using mean of embeddings)
        import numpy as np
        scores = [np.mean(emb) for emb in embeddings]
        
        # Sort words by importance score
        word_scores = list(zip(words, scores))
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        keywords = word_scores[:req.max_keywords]
        return {
            "keywords": [
                {"word": word, "score": round(float(score), 3)}
                for word, score in keywords
            ]
        }
        
    except Exception as e:
        logger.exception("Keyword extraction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-text")
async def generate_text(req: TextGenInput):
    """Generate text based on a prompt."""
    try:
        from transformers import pipeline as gen_pipeline
        
        generator = gen_pipeline(
            "text-generation",
            model="gpt2",
            device=-1  # Use CPU. Set to 0 for GPU if available
        )
        
        result = generator(
            req.prompt,
            max_length=req.max_length,
            temperature=req.temperature,
            num_return_sequences=1,
            do_sample=True
        )[0]
        
        return {
            "generated_text": result["generated_text"],
            "prompt": req.prompt,
            "settings": {
                "max_length": req.max_length,
                "temperature": req.temperature
            }
        }
        
    except Exception as e:
        logger.exception("Text generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-entities")
async def extract_entities(req: EntityExtractionIn):
    """Extract named entities (people, organizations, locations) from text."""
    try:
        from transformers import pipeline as ner_pipeline
        
        # Initialize NER pipeline
        ner = ner_pipeline("ner", aggregation_strategy="simple")
        
        # Get entities
        entities = ner(req.text)
        
        # Filter and format results
        filtered_entities = []
        for ent in entities:
            # Extract entity type (e.g., PER from B-PER or I-PER)
            ent_type = ent["entity_group"]
            if ent_type in req.include_types:
                filtered_entities.append({
                    "text": ent["word"],
                    "type": ent_type,
                    "score": round(float(ent["score"]), 3),
                    "start": ent["start"],
                    "end": ent["end"]
                })
        
        return {
            "entities": filtered_entities,
            "text": req.text
        }
        
    except Exception as e:
        logger.exception("Entity extraction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
