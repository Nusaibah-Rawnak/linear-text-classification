from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from pa1 import load_data
import os

# --- Models stored here after training ---
models = {}

# --- Train on startup using lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    base = os.path.dirname(os.path.abspath(__file__))

    # Task A: Sentiment
    texts_a, labels_a = load_data(
        os.path.join(base, "synsem0.txt"),
        os.path.join(base, "synsem1.txt")
    )
    vec_a = CountVectorizer()
    X_a = vec_a.fit_transform(texts_a)
    clf_a = LogisticRegression(max_iter=1000, random_state=123)
    clf_a.fit(X_a, labels_a)
    models["sentiment"] = (vec_a, clf_a)

    # Task B: Alliteration
    texts_b, labels_b = load_data(
        os.path.join(base, "morphphon0.txt"),
        os.path.join(base, "morphphon1.txt")
    )
    vec_b = CountVectorizer()
    X_b = vec_b.fit_transform(texts_b)
    clf_b = LogisticRegression(max_iter=1000, random_state=123)
    clf_b.fit(X_b, labels_b)
    models["alliteration"] = (vec_b, clf_b)

    print("Models trained and ready!")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request schema ---
class PredictRequest(BaseModel):
    sentence: str
    task: str  # "sentiment" or "alliteration"

# --- Predict endpoint ---
@app.post("/predict")
def predict(req: PredictRequest):
    if req.task not in models:
        raise HTTPException(status_code=400, detail="task must be 'sentiment' or 'alliteration'")

    vec, clf = models[req.task]
    X = vec.transform([req.sentence])
    label_idx = clf.predict(X)[0]
    confidence = float(clf.predict_proba(X)[0][label_idx])

    if req.task == "sentiment":
        label = "positive" if label_idx == 1 else "negative"
    else:
        label = "alliterative" if label_idx == 1 else "not alliterative"

    return {"label": label, "confidence": round(confidence, 4)}

# --- Health check ---
@app.get("/health")
def health():
    return {"status": "ok"}