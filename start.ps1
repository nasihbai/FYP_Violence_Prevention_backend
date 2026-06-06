# ============================================================
# EYES of Soteria — Full Stack Startup
# ============================================================
# Run from the backend folder:
#   .\start.ps1
#
# What it does:
#   1. Starts PostgreSQL + Adminer + pgAdmin via Docker
#   2. Waits until the database is healthy
#   3. Opens the Flask backend in a new terminal window/tab
#   4. Opens the Vue frontend in a new terminal window/tab
#
# Requirements:
#   - Docker Desktop running
#   - Python venv at .\venv\
#   - Violence_Prevention_Frontend folder next to this repo
# ============================================================

$ErrorActionPreference = "Stop"
$root     = $PSScriptRoot
$frontend = Resolve-Path (Join-Path $root "..\Violence_Prevention_Frontend")
$python   = Join-Path $root "venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "ERROR: venv not found at $python" -ForegroundColor Red
    Write-Host "       Create it with: python -m venv venv && venv\Scripts\pip install -r requirements.txt"
    exit 1
}

Write-Host ""
Write-Host "  EYES of Soteria — Starting all services" -ForegroundColor Cyan
Write-Host "  ========================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Docker services ────────────────────────────────
Write-Host "[1/3] Starting database containers..." -ForegroundColor Yellow
docker compose up db adminer pgadmin -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: docker compose failed. Is Docker Desktop running?" -ForegroundColor Red
    exit 1
}

# Wait for PostgreSQL to become healthy (up to 40s)
Write-Host "      Waiting for PostgreSQL to be ready..." -ForegroundColor DarkGray
$ready = $false
for ($i = 0; $i -lt 20; $i++) {
    $status = docker compose ps db 2>&1 | Select-String "healthy"
    if ($status) { $ready = $true; break }
    Start-Sleep 2
}
if (-not $ready) {
    Write-Host "WARNING: DB health check timed out — backend may fail to connect initially." -ForegroundColor DarkYellow
}
Write-Host "      Database ready." -ForegroundColor Green

# ── Step 2: Backend (Flask) ────────────────────────────────
Write-Host "[2/3] Starting Flask backend..." -ForegroundColor Yellow
$beCmd = "Set-Location '$root'; Write-Host 'Backend — http://localhost:5000' -ForegroundColor Cyan; .\venv\Scripts\python.exe run_detection.py --web"

# ── Step 3: Frontend (Vue) ─────────────────────────────────
Write-Host "[3/3] Starting Vue frontend..." -ForegroundColor Yellow
$feCmd = "Set-Location '$frontend'; Write-Host 'Frontend — http://localhost:3001' -ForegroundColor Cyan; pnpm dev"

# Try Windows Terminal first (gives nice tabs), then fall back to separate windows
if (Get-Command wt -ErrorAction SilentlyContinue) {
    wt --window 0 new-tab --title "Backend (Flask)"   powershell -NoExit -Command $beCmd
    Start-Sleep 1
    wt --window 0 new-tab --title "Frontend (Vue)"    powershell -NoExit -Command $feCmd
} else {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $beCmd
    Start-Sleep 1
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $feCmd
}

# ── Summary ────────────────────────────────────────────────
Write-Host ""
Write-Host "  All services starting. Open these URLs:" -ForegroundColor Green
Write-Host ""
Write-Host "    App       http://localhost:3001   (admin@example.com / admin123)" -ForegroundColor White
Write-Host "    Backend   http://localhost:5000/health" -ForegroundColor White
Write-Host "    Adminer   http://localhost:8080   (Server: db, User: violence_user)" -ForegroundColor White
Write-Host "    pgAdmin   http://localhost:5050   (admin@example.com / admin123)" -ForegroundColor White
Write-Host ""
Write-Host "  To stop:  docker compose down  (then Ctrl+C in each window)" -ForegroundColor DarkGray
Write-Host ""
