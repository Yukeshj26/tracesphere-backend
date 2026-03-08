"""
Approvals routes — CRUD for Firestore 'approvals' collection
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime
from models.schemas import ApprovalCreate, ApprovalUpdate
from services.firebase import get_db

router = APIRouter()


@router.get("/")
def get_approvals(
    status:     Optional[str] = Query(None),
    priority:   Optional[str] = Query(None),
    department: Optional[str] = Query(None),
):
    db   = get_db()
    docs = db.collection("approvals").stream()
    items = []
    for doc in docs:
        data = doc.to_dict()
        data["firestoreId"] = doc.id
        if status     and data.get("status")     != status:     continue
        if priority   and data.get("priority")   != priority:   continue
        if department and data.get("department") != department: continue
        items.append(data)
    return items


@router.get("/{approval_id}")
def get_approval(approval_id: str):
    db  = get_db()
    doc = db.collection("approvals").document(approval_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Approval not found")
    data = doc.to_dict()
    data["firestoreId"] = doc.id
    return data


@router.post("/", status_code=201)
def create_approval(approval: ApprovalCreate):
    db   = get_db()
    data = approval.dict(exclude_none=True)
    data["createdAt"] = datetime.utcnow().isoformat()
    ref  = db.collection("approvals").add(data)
    return {"firestoreId": ref[1].id, **data}


@router.put("/{approval_id}")
def update_approval(approval_id: str, updates: ApprovalUpdate):
    db  = get_db()
    ref = db.collection("approvals").document(approval_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Approval not found")
    data = {k: v for k, v in updates.dict().items() if v is not None}
    data["updatedAt"] = datetime.utcnow().isoformat()
    ref.update(data)
    updated = ref.get().to_dict()
    updated["firestoreId"] = approval_id
    return updated


@router.delete("/{approval_id}")
def delete_approval(approval_id: str):
    db  = get_db()
    ref = db.collection("approvals").document(approval_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Approval not found")
    ref.delete()
    return {"message": f"Approval {approval_id} deleted"}


@router.patch("/{approval_id}/approve")
def approve(approval_id: str, approved_by: str):
    db  = get_db()
    ref = db.collection("approvals").document(approval_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Approval not found")
    ref.update({
        "status":     "approved",
        "approvedBy": approved_by,
        "updatedAt":  datetime.utcnow().isoformat(),
    })
    return {"message": "Approved", "approvedBy": approved_by}


@router.patch("/{approval_id}/reject")
def reject(approval_id: str, reason: str = ""):
    db  = get_db()
    ref = db.collection("approvals").document(approval_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Approval not found")
    ref.update({
        "status":    "rejected",
        "reason":    reason,
        "updatedAt": datetime.utcnow().isoformat(),
    })
    return {"message": "Rejected"}
