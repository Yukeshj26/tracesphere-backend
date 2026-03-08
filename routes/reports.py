"""
Reports routes — generate PDF and CSV reports server-side
"""
import io
import csv
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from datetime import datetime
from services.firebase import get_db
from services.ml_model import get_model

router = APIRouter()


def _get_assets():
    db = get_db()
    return [doc.to_dict() for doc in db.collection("assets").stream()]

def _get_procurement():
    db = get_db()
    return [doc.to_dict() for doc in db.collection("procurement").stream()]


# ── CSV helpers ───────────────────────────────────────────────────────────────
def make_csv_response(filename: str, headers: list, rows: list):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ── Inventory Summary ─────────────────────────────────────────────────────────
@router.get("/inventory/csv")
def inventory_csv():
    assets = _get_assets()
    headers = ["Asset ID","Name","Category","Location","Quantity","Unit","Status","Department","Cost (Rs)","Purchase Date"]
    rows    = [[
        a.get("assetId",""), a.get("name",""), a.get("category",""), a.get("location",""),
        a.get("quantity",""), a.get("unit",""), a.get("status",""),
        a.get("department",""), a.get("cost",""), a.get("purchaseDate","")
    ] for a in assets]
    return make_csv_response("tracesphere-inventory-summary.csv", headers, rows)


# ── Low Stock Report ──────────────────────────────────────────────────────────
@router.get("/lowstock/csv")
def lowstock_csv():
    assets   = _get_assets()
    low      = [a for a in assets if int(a.get("quantity",0) or 0) <= int(a.get("minQuantity",0) or 0)]
    headers  = ["Asset ID","Name","Category","Location","Current Qty","Min Qty","Unit","Department"]
    rows     = [[
        a.get("assetId",""), a.get("name",""), a.get("category",""), a.get("location",""),
        a.get("quantity",""), a.get("minQuantity",""), a.get("unit",""), a.get("department","")
    ] for a in low]
    return make_csv_response("tracesphere-lowstock.csv", headers, rows)


# ── Financial Summary ─────────────────────────────────────────────────────────
@router.get("/financial/csv")
def financial_csv():
    assets  = _get_assets()
    total   = sum((float(a.get("cost",0) or 0)) * (int(a.get("quantity",0) or 0)) for a in assets)
    headers = ["Asset ID","Name","Category","Quantity","Unit Cost (Rs)","Total Value (Rs)","Purchase Date"]
    rows    = [[
        a.get("assetId",""), a.get("name",""), a.get("category",""),
        a.get("quantity",""),
        a.get("cost",""),
        round(float(a.get("cost",0) or 0) * int(a.get("quantity",0) or 0), 2),
        a.get("purchaseDate","")
    ] for a in assets]
    rows.append(["","","","","GRAND TOTAL", round(total, 2), ""])
    return make_csv_response("tracesphere-financial-summary.csv", headers, rows)


# ── ML Forecast Report ────────────────────────────────────────────────────────
@router.get("/forecast/csv")
def forecast_csv():
    assets      = _get_assets()
    procurement = _get_procurement()
    results     = get_model().predict(assets, procurement)
    headers = [
        "Asset ID","Name","Category","Department","Current Qty","Min Qty",
        "Pred Qty 30d","Pred Qty 60d","Monthly Consumption","Days Until Low",
        "Reorder Qty","Est Cost","Risk","Confidence","R2 Score","Season","Proc Boosted"
    ]
    rows = [[
        r["assetId"], r["name"], r["category"], r["department"],
        r["currentQty"], r["minQty"], r["predictedQty30"], r["predictedQty60"],
        r["monthlyConsumption"], r["daysUntilLow"] if r["daysUntilLow"] != 999 else "Safe",
        r["reorderQty"], r["estimatedCost"], r["risk"], r["confidence"],
        r["r2Score"], r["seasonLabel"], "Yes" if r["procBoosted"] else "No"
    ] for r in results]
    return make_csv_response("tracesphere-ml-forecast.csv", headers, rows)


# ── Reorder Report (same as dashboard button) ─────────────────────────────────
@router.get("/reorder/csv")
def reorder_csv():
    assets  = _get_assets()
    low     = [a for a in assets if int(a.get("quantity",0) or 0) <= int(a.get("minQuantity",0) or 0)]
    headers = ["Asset ID","Item Name","Category","Department","Location","Current Qty","Min Qty","Reorder Qty","Unit Cost (Rs)","Total Est. (Rs)","Status"]
    rows = []
    for a in low:
        curr       = int(a.get("quantity",0) or 0)
        mn         = int(a.get("minQuantity",1) or 1)
        cost       = float(a.get("cost",0) or 0)
        reorder    = max(mn, mn * 2 - curr)
        rows.append([
            a.get("assetId",""), a.get("name",""), a.get("category",""),
            a.get("department",""), a.get("location",""),
            curr, mn, reorder,
            cost, round(reorder * cost, 2),
            "OUT OF STOCK" if curr == 0 else "LOW STOCK"
        ])
    total_row = ["","","","","","","","","TOTAL",
                 round(sum(float(r[9]) for r in rows), 2), ""]
    rows.append(total_row)
    return make_csv_response("tracesphere-reorder-report.csv", headers, rows)


# ── All reports summary ───────────────────────────────────────────────────────
@router.get("/summary")
def reports_summary():
    assets      = _get_assets()
    procurement = _get_procurement()
    low         = [a for a in assets if int(a.get("quantity",0) or 0) <= int(a.get("minQuantity",0) or 0)]
    total_value = sum((float(a.get("cost",0) or 0)) * int(a.get("quantity",0) or 0) for a in assets)
    forecast    = get_model().predict(assets, procurement)
    return {
        "generatedAt":        datetime.utcnow().isoformat(),
        "totalAssets":        len(assets),
        "lowStockCount":      len(low),
        "totalInventoryValue": round(total_value, 2),
        "procurementOrders":  len(procurement),
        "mlForecast": {
            "critical": sum(1 for r in forecast if r["risk"] == "critical"),
            "high":     sum(1 for r in forecast if r["risk"] == "high"),
            "medium":   sum(1 for r in forecast if r["risk"] == "medium"),
            "low":      sum(1 for r in forecast if r["risk"] == "low"),
        }
    }
