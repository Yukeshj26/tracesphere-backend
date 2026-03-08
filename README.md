# TraceSphere Backend — Python FastAPI

## Setup

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Add Firebase Service Account Key
- Go to: Firebase Console → Project Settings → Service Accounts
- Click "Generate New Private Key"
- Save the file as `serviceAccountKey.json` inside the `backend/` folder

### 3. Run the server
```bash
uvicorn main:app --reload --port 8000
```

### 4. Open API docs
```
http://localhost:8000/docs
```

---

## API Endpoints

### Assets
| Method | URL | Description |
|--------|-----|-------------|
| GET    | /api/assets/ | Get all assets (with filters) |
| GET    | /api/assets/{id} | Get single asset |
| POST   | /api/assets/ | Create asset |
| PUT    | /api/assets/{id} | Update asset |
| DELETE | /api/assets/{id} | Delete asset |
| PATCH  | /api/assets/{id}/quantity | Quick quantity update |
| GET    | /api/assets/stats/summary | Dashboard KPI stats |

### Procurement
| Method | URL | Description |
|--------|-----|-------------|
| GET    | /api/procurement/ | Get all orders |
| POST   | /api/procurement/ | Create order |
| PUT    | /api/procurement/{id} | Update order |
| PATCH  | /api/procurement/{id}/status | Update status |
| DELETE | /api/procurement/{id} | Delete order |

### Approvals
| Method | URL | Description |
|--------|-----|-------------|
| GET    | /api/approvals/ | Get all approvals |
| POST   | /api/approvals/ | Create approval request |
| PATCH  | /api/approvals/{id}/approve | Approve request |
| PATCH  | /api/approvals/{id}/reject  | Reject request |

### ML Forecast
| Method | URL | Description |
|--------|-----|-------------|
| GET    | /api/forecast/ | Full ML forecast for all assets |
| GET    | /api/forecast/summary | Risk counts + budget estimate |
| GET    | /api/forecast/critical | Only critical + high risk items |
| GET    | /api/forecast/seasonal-calendar | CIT academic calendar multipliers |

### Reports (CSV downloads)
| Method | URL | Description |
|--------|-----|-------------|
| GET    | /api/reports/inventory/csv | Inventory summary CSV |
| GET    | /api/reports/lowstock/csv  | Low stock report CSV |
| GET    | /api/reports/financial/csv | Financial summary CSV |
| GET    | /api/reports/forecast/csv  | ML forecast CSV |
| GET    | /api/reports/reorder/csv   | Reorder report CSV |
| GET    | /api/reports/summary       | All stats summary JSON |

---

## ML Forecast Algorithm

```
For each asset:
  1. Build 12-month synthetic history from purchase date + category rate
  2. Run OLS Linear Regression → slope (depletion rate) + R² (confidence)
  3. Blend regression with category model: R² × regression + (1-R²) × rate model
  4. Apply CIT academic calendar seasonal multiplier (Feb/Nov = 1.6x peak)
  5. Apply procurement frequency boost (frequently ordered = higher demand)
  6. Predict quantity at 30 and 60 days
  7. Calculate days until stock hits minimum threshold
  8. Confidence score = R² score + data points + proximity to minimum
  9. Risk: Critical (≤7d) | High (≤20d) | Medium (≤35d) | Low (safe)
  10. Seasonal override: upcoming exam season bumps risk up one level
```

## Project Structure
```
backend/
├── main.py                  # FastAPI app entry point
├── requirements.txt
├── serviceAccountKey.json   # ← YOU ADD THIS (not in git)
├── routes/
│   ├── assets.py
│   ├── procurement.py
│   ├── approvals.py
│   ├── forecast.py
│   ├── reports.py
│   └── auth.py
├── services/
│   ├── firebase.py          # Firestore connection
│   └── ml_forecast.py       # ML engine (OLS + Seasonality)
└── models/
    └── schemas.py           # Pydantic request/response models
```
