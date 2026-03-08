// src/services/mlService.js
// Calls the Python ML backend for forecast data

const ML_API = process.env.REACT_APP_ML_API || 'http://localhost:8000';

/**
 * Run full ML forecast — returns all assets with risk, predictions, confidence
 */
export async function fetchMLForecast(filter = {}) {
  const params = new URLSearchParams();
  if (filter.risk)       params.append('risk',       filter.risk);
  if (filter.department) params.append('department', filter.department);

  const res  = await fetch(`${ML_API}/api/forecast/?${params}`);
  if (!res.ok) throw new Error('ML forecast failed');
  return res.json();
}

/**
 * Get just the summary counts (critical/high/medium/low + budget)
 * Used for dashboard KPI cards
 */
export async function fetchForecastSummary() {
  const res = await fetch(`${ML_API}/api/forecast/summary`);
  if (!res.ok) throw new Error('Forecast summary failed');
  return res.json();
}

/**
 * Full analytics report — department breakdown, category breakdown,
 * 3-month budget forecast, anomalies, top risk items
 */
export async function fetchReportAnalytics() {
  const res = await fetch(`${ML_API}/api/reports/summary`);
  if (!res.ok) throw new Error('Report analytics failed');
  return res.json();
}

/**
 * Download ML forecast as CSV from backend
 */
export function downloadForecastCSV() {
  window.open(`${ML_API}/api/reports/forecast/csv`, '_blank');
}
