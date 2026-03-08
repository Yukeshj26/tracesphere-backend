"""
Pydantic models — request/response schemas for all collections
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import date


# ── Asset ─────────────────────────────────────────────────────────────────────
class AssetBase(BaseModel):
    assetId:      str
    name:         str
    category:     str
    location:     str
    quantity:     int
    minQuantity:  int
    unit:         str
    status:       Literal["available", "issued", "maintenance", "disposed"] = "available"
    department:   Optional[str] = None
    cost:         Optional[float] = None
    purchaseDate: Optional[str]  = None
    description:  Optional[str]  = None

class AssetCreate(AssetBase):
    pass

class AssetUpdate(BaseModel):
    name:         Optional[str]   = None
    category:     Optional[str]   = None
    location:     Optional[str]   = None
    quantity:     Optional[int]   = None
    minQuantity:  Optional[int]   = None
    unit:         Optional[str]   = None
    status:       Optional[str]   = None
    department:   Optional[str]   = None
    cost:         Optional[float] = None
    purchaseDate: Optional[str]   = None
    description:  Optional[str]   = None

class AssetOut(AssetBase):
    firestoreId: str


# ── Procurement ───────────────────────────────────────────────────────────────
class ProcurementBase(BaseModel):
    poNumber:     str
    itemName:     str
    category:     str
    department:   str
    supplier:     str
    quantity:     int
    unit:         str
    unitCost:     float
    totalCost:    float
    status:       Literal["pending", "approved", "ordered", "received", "cancelled"] = "pending"
    requestedBy:  Optional[str] = None
    orderDate:    Optional[str] = None
    expectedDate: Optional[str] = None
    notes:        Optional[str] = None

class ProcurementCreate(ProcurementBase):
    pass

class ProcurementUpdate(BaseModel):
    status:       Optional[str]   = None
    supplier:     Optional[str]   = None
    unitCost:     Optional[float] = None
    totalCost:    Optional[float] = None
    expectedDate: Optional[str]   = None
    notes:        Optional[str]   = None

class ProcurementOut(ProcurementBase):
    firestoreId: str


# ── Approval ──────────────────────────────────────────────────────────────────
class ApprovalBase(BaseModel):
    reqId:         str
    itemName:      str
    category:      str
    department:    str
    requestedBy:   str
    quantity:      int
    unit:          str
    estimatedCost: float
    priority:      Literal["low", "medium", "high"] = "medium"
    status:        Literal["pending", "approved", "rejected"] = "pending"
    reason:        Optional[str] = None
    approvedBy:    Optional[str] = None
    date:          Optional[str] = None

class ApprovalCreate(ApprovalBase):
    pass

class ApprovalUpdate(BaseModel):
    status:     Optional[str] = None
    approvedBy: Optional[str] = None
    reason:     Optional[str] = None

class ApprovalOut(ApprovalBase):
    firestoreId: str


# ── ML Forecast Result ────────────────────────────────────────────────────────
class ForecastItem(BaseModel):
    assetId:            str
    name:               str
    category:           str
    department:         str
    currentQty:         int
    minQty:             int
    predictedQty30:     int
    predictedQty60:     int
    daysUntilLow:       int
    monthlyConsumption: float
    reorderQty:         int
    estimatedCost:      float
    confidence:         int
    risk:               Literal["critical", "high", "medium", "low"]
    r2Score:            float
    seasonMultiplier:   float
    seasonLabel:        str
    procBoosted:        bool
