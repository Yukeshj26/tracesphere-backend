"""
Procurement routes — CRUD for Firestore 'procurement' collection
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from models.schemas import ProcurementCreate, ProcurementUpdate
from services.firebase import get_db

router = APIRouter()


@router.get("/", response_model=List[dict])
def get_procurement(
    status:     Optional[str] = Query(None),
    department: Optional[str] = Query(None),
):
    db   = get_db()
    docs = db.collection("procurement").stream()
    orders = []
    for doc in docs:
        data = doc.to_dict()
        data["firestoreId"] = doc.id
        if status     and data.get("status")     != status:     continue
        if department and data.get("department") != department: continue
        orders.append(data)
    return orders


@router.get("/{order_id}", response_model=dict)
def get_order(order_id: str):
    db  = get_db()
    doc = db.collection("procurement").document(order_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Order not found")
    data = doc.to_dict()
    data["firestoreId"] = doc.id
    return data


@router.post("/", response_model=dict, status_code=201)
def create_order(order: ProcurementCreate):
    db   = get_db()
    data = order.dict(exclude_none=True)
    data["createdAt"] = datetime.utcnow().isoformat()
    ref  = db.collection("procurement").add(data)
    return {"firestoreId": ref[1].id, **data}


@router.put("/{order_id}", response_model=dict)
def update_order(order_id: str, updates: ProcurementUpdate):
    db  = get_db()
    ref = db.collection("procurement").document(order_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Order not found")
    data = {k: v for k, v in updates.dict().items() if v is not None}
    data["updatedAt"] = datetime.utcnow().isoformat()
    ref.update(data)
    updated = ref.get().to_dict()
    updated["firestoreId"] = order_id
    return updated


@router.delete("/{order_id}")
def delete_order(order_id: str):
    db  = get_db()
    ref = db.collection("procurement").document(order_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Order not found")
    ref.delete()
    return {"message": f"Order {order_id} deleted"}


@router.patch("/{order_id}/status")
def update_status(order_id: str, status: str):
    valid = ["pending", "approved", "ordered", "received", "cancelled"]
    if status not in valid:
        raise HTTPException(status_code=400, detail=f"Status must be one of {valid}")
    db  = get_db()
    ref = db.collection("procurement").document(order_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Order not found")
    ref.update({"status": status, "updatedAt": datetime.utcnow().isoformat()})
    return {"message": "Status updated", "status": status}
