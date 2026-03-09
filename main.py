"""
TraceSphere Backend — FastAPI + Firebase Admin SDK
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import assets, procurement, approvals, forecast, reports, auth

app = FastAPI(
    title="TraceSphere API",
    description="Inventory Management Backend — Chennai Institute of Technology",
    version="1.0.0"
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,        prefix="/api/auth",        tags=["Auth"])
app.include_router(assets.router,      prefix="/api/assets",      tags=["Assets"])
app.include_router(procurement.router, prefix="/api/procurement",  tags=["Procurement"])
app.include_router(approvals.router,   prefix="/api/approvals",   tags=["Approvals"])
app.include_router(forecast.router,    prefix="/api/forecast",    tags=["ML Forecast"])
app.include_router(reports.router,     prefix="/api/reports",     tags=["Reports"])

@app.on_event("startup")
async def startup_event():
    """
    On startup:
      - If saved model exists on disk → load it (instant, no Firestore reads)
      - If no saved model → train from Firestore and save to disk
    This prevents retraining on every Render wake-up and saves Firestore quota.
    """
    try:
        from services.ml_model import get_model
        model = get_model()

        if os.path.exists("models/rf_model.pkl"):
            # Model already saved — just load it, no Firestore reads needed
            print("[startup] Saved model found — loading from disk (skipping retrain)")
            model.load("models")
            print("[startup] ML model ready from cache")
        else:
            # First run — train from Firestore and save
            print("[startup] No saved model — fetching data and training...")
            from services.firebase import get_db
            db          = get_db()
            assets_data = [doc.to_dict() for doc in db.collection("assets").stream()]
            proc_data   = [doc.to_dict() for doc in db.collection("procurement").stream()]
            model.train(assets_data, proc_data, verbose=True)
            model.save("models")
            print(f"[startup] ML model trained and saved — {len(assets_data)} assets")

    except Exception as e:
        print(f"[startup] ML startup skipped: {e}")

@app.get("/")
def root():
    return {"app": "TraceSphere API", "version": "1.0.0", "status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}
