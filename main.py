"""
TraceSphere Backend — FastAPI + Firebase Admin SDK
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import assets, procurement, approvals, forecast, reports, auth

app = FastAPI(
    title="TraceSphere API",
    description="Inventory Management Backend — Chennai Institute of Technology",
    version="1.0.0"
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow all origins so localhost:3000, :5173, :5174, firebase hosting all work
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # must be False when allow_origins=["*"]
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
    """Pre-train ML model on startup so first request is instant."""
    try:
        print("[startup] Pre-training ML model...")
        from services.firebase import get_db
        from services.ml_model import get_model
        db          = get_db()
        assets_data = [doc.to_dict() for doc in db.collection("assets").stream()]
        proc_data   = [doc.to_dict() for doc in db.collection("procurement").stream()]
        model = get_model()
        model.train(assets_data, proc_data, verbose=True)
        model.save("models")
        print(f"[startup] ML model ready — {len(assets_data)} assets trained")
    except Exception as e:
        print(f"[startup] ML pre-train skipped: {e}")

@app.get("/")
def root():
    return {"app": "TraceSphere API", "version": "1.0.0", "status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}
