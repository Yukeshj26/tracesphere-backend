"""
Assets routes — CRUD for Firestore 'assets' collection
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from models.schemas import AssetCreate, AssetUpdate, AssetOut
from services.firebase import get_db

router = APIRouter()


@router.get("/", response_model=List[dict])
def get_assets(
    department: Optional[str] = Query(None),
    category:   Optional[str] = Query(None),
    status:     Optional[str] = Query(None),
    low_stock:  bool = Query(False, description="Only return items below min quantity"),
):
    """Get all assets with optional filters."""
    db   = get_db()
    ref  = db.collection("assets")
    docs = ref.stream()

    assets = []
    for doc in docs:
        data = doc.to_dict()
        data["firestoreId"] = doc.id

        # Apply filters
        if department and data.get("department") != department:
            continue
        if category and data.get("category") != category:
            continue
        if status and data.get("status") != status:
            continue
        if low_stock:
            qty = int(data.get("quantity", 0) or 0)
            min_qty = int(data.get("minQuantity", 0) or 0)
            if qty > min_qty:
                continue

        assets.append(data)

    return assets


@router.get("/{asset_id}", response_model=dict)
def get_asset(asset_id: str):
    """Get single asset by Firestore document ID."""
    db  = get_db()
    doc = db.collection("assets").document(asset_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Asset not found")
    data = doc.to_dict()
    data["firestoreId"] = doc.id
    return data


@router.post("/", response_model=dict, status_code=201)
def create_asset(asset: AssetCreate):
    """Create a new asset."""
    db   = get_db()
    data = asset.dict(exclude_none=True)
    data["createdAt"] = datetime.utcnow().isoformat()

    # Check for duplicate assetId
    existing = db.collection("assets").where("assetId", "==", asset.assetId).limit(1).stream()
    if any(True for _ in existing):
        raise HTTPException(status_code=400, detail=f"Asset ID '{asset.assetId}' already exists")

    ref = db.collection("assets").add(data)
    return {"firestoreId": ref[1].id, **data}


@router.put("/{asset_id}", response_model=dict)
def update_asset(asset_id: str, updates: AssetUpdate):
    """Update an existing asset."""
    db  = get_db()
    ref = db.collection("assets").document(asset_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Asset not found")

    data = {k: v for k, v in updates.dict().items() if v is not None}
    data["updatedAt"] = datetime.utcnow().isoformat()
    ref.update(data)

    updated = ref.get().to_dict()
    updated["firestoreId"] = asset_id
    return updated


@router.delete("/{asset_id}")
def delete_asset(asset_id: str):
    """Delete an asset."""
    db  = get_db()
    ref = db.collection("assets").document(asset_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Asset not found")
    ref.delete()
    return {"message": f"Asset {asset_id} deleted successfully"}


@router.patch("/{asset_id}/quantity")
def update_quantity(asset_id: str, quantity: int):
    """Quick update just the quantity of an asset."""
    if quantity < 0:
        raise HTTPException(status_code=400, detail="Quantity cannot be negative")
    db  = get_db()
    ref = db.collection("assets").document(asset_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Asset not found")
    ref.update({"quantity": quantity, "updatedAt": datetime.utcnow().isoformat()})
    return {"message": "Quantity updated", "quantity": quantity}


@router.get("/stats/summary")
def get_stats():
    """Dashboard KPI stats."""
    db   = get_db()
    docs = list(db.collection("assets").stream())

    total        = len(docs)
    low_stock    = 0
    maintenance  = 0
    total_value  = 0.0

    for doc in docs:
        d   = doc.to_dict()
        qty = int(d.get("quantity", 0) or 0)
        mn  = int(d.get("minQuantity", 0) or 0)
        if qty <= mn:
            low_stock += 1
        if d.get("status") == "maintenance":
            maintenance += 1
        total_value += float(d.get("cost", 0) or 0) * qty

    return {
        "totalAssets":   total,
        "lowStock":      low_stock,
        "maintenance":   maintenance,
        "totalValue":    round(total_value, 2),
    }
