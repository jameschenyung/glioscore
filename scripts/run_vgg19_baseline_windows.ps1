#Requires -Version 5.1
<#
.SYNOPSIS
  One-shot Windows runner for the VGG19-RDS XGBoost baseline (manager metrics).

.DESCRIPTION
  - Uses an existing glioscore checkout if present; otherwise clones
    https://github.com/jameschenyung/glioscore.git
  - Checks out branch cursor/vgg19-rds-baseline-92d1 (PR #2) when needed
  - Creates .venv, installs requirements (+ pyreadr, xgboost)
  - Runs train_vgg19_rds_baseline.py against the canonical Downloads RDS
  - Prints and displays outputs/xgboost_vgg19_rds_metrics.json

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\run_vgg19_baseline_windows.ps1

.NOTES
  Canonical RDS (leave in place; no copy into data/):
    C:\Users\james\Downloads\lgggbm_cnn_byRegion_VGG19.rds
#>

$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/jameschenyung/glioscore.git"
$Branch = "cursor/vgg19-rds-baseline-92d1"
$RdsPath = "C:\Users\james\Downloads\lgggbm_cnn_byRegion_VGG19.rds"
$DefaultCloneDir = Join-Path $env:USERPROFILE "glioscore"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-IsGlioscoreRepo([string]$Path) {
    if (-not (Test-Path -LiteralPath (Join-Path $Path ".git"))) { return $false }
    $marker = Join-Path $Path "scripts\train_vgg19_rds_baseline.py"
    $altMarker = Join-Path $Path "scripts\train_xgboost.py"
    return (Test-Path -LiteralPath $marker) -or (Test-Path -LiteralPath $altMarker)
}

function Resolve-RepoRoot {
    # 1) Script lives inside scripts/ of a checkout
    if ($PSScriptRoot) {
        $fromScript = Split-Path -Parent $PSScriptRoot
        if (Test-IsGlioscoreRepo $fromScript) { return $fromScript }
    }
    # 2) Current working directory is the repo (or a subfolder)
    $cwd = (Get-Location).Path
    if (Test-IsGlioscoreRepo $cwd) { return $cwd }
    $parent = Split-Path -Parent $cwd
    if ($parent -and (Test-IsGlioscoreRepo $parent)) { return $parent }
    # 3) Common clone location
    if (Test-IsGlioscoreRepo $DefaultCloneDir) { return $DefaultCloneDir }
    return $null
}

# --- RDS must exist before we spend time on venv ---
if (-not (Test-Path -LiteralPath $RdsPath)) {
    throw @"
RDS file not found:
  $RdsPath
Leave the manager file at that path (no copy into data/ required), then re-run.
"@
}

Write-Step "Locating glioscore repo"
$RepoRoot = Resolve-RepoRoot
if (-not $RepoRoot) {
    Write-Host "No local checkout found; cloning to $DefaultCloneDir"
    if (Test-Path -LiteralPath $DefaultCloneDir) {
        throw "Directory exists but is not a glioscore checkout: $DefaultCloneDir"
    }
    git clone $RepoUrl $DefaultCloneDir
    $RepoRoot = $DefaultCloneDir
}
Write-Host "Repo: $RepoRoot"
Set-Location -LiteralPath $RepoRoot

Write-Step "Ensuring branch $Branch"
git fetch origin $Branch
if ($LASTEXITCODE -ne 0) {
    git fetch origin
}
$current = (git rev-parse --abbrev-ref HEAD).Trim()
if ($current -ne $Branch) {
    git show-ref --verify --quiet "refs/heads/$Branch"
    if ($LASTEXITCODE -eq 0) {
        git checkout $Branch
    } else {
        git checkout -B $Branch "origin/$Branch"
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to check out $Branch. Is the remote branch pushed?"
    }
}
git pull --ff-only origin $Branch
Write-Host "On branch: $((git rev-parse --abbrev-ref HEAD).Trim()) @ $((git rev-parse --short HEAD).Trim())"

# Prefer py / python3 / python on Windows
$PythonCmd = $null
foreach ($cand in @("py", "python", "python3")) {
    $cmd = Get-Command $cand -ErrorAction SilentlyContinue
    if ($cmd) {
        $PythonCmd = $cand
        break
    }
}
if (-not $PythonCmd) {
    throw "Python not found on PATH. Install Python 3.10+ and re-run."
}
Write-Host "Python launcher: $PythonCmd"

Write-Step "Creating / refreshing .venv"
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    if ($PythonCmd -eq "py") {
        & py -3 -m venv $VenvDir
    } else {
        & $PythonCmd -m venv $VenvDir
    }
}
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Failed to create venv at $VenvDir"
}

Write-Step "Installing baseline deps (numpy/pandas/sklearn/xgboost/pyreadr)"
# Lean install for this baseline only — avoids pulling torch/timm from full requirements.txt.
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install `
    "numpy>=1.26,<3" `
    "pandas>=2.1,<4" `
    "scikit-learn>=1.4,<2" `
    "xgboost>=2.0,<4" `
    "pyreadr>=0.5,<1"

Write-Step "Training VGG19-RDS baseline (patient-level split)"
$TrainScript = Join-Path $RepoRoot "scripts\train_vgg19_rds_baseline.py"
if (-not (Test-Path -LiteralPath $TrainScript)) {
    throw "Missing $TrainScript — is branch $Branch checked out?"
}
& $VenvPython $TrainScript --rds $RdsPath
if ($LASTEXITCODE -ne 0) {
    throw "Baseline training failed with exit code $LASTEXITCODE"
}

$MetricsPath = Join-Path $RepoRoot "outputs\xgboost_vgg19_rds_metrics.json"
if (-not (Test-Path -LiteralPath $MetricsPath)) {
    throw "Expected metrics file was not written: $MetricsPath"
}

Write-Step "Metrics file"
Write-Host $MetricsPath -ForegroundColor Green
Write-Host ""
Write-Host "----- xgboost_vgg19_rds_metrics.json -----" -ForegroundColor Yellow
Get-Content -LiteralPath $MetricsPath -Raw
Write-Host "----- end metrics -----" -ForegroundColor Yellow
Write-Host ""
Write-Host "Done. Share the JSON above with your manager." -ForegroundColor Green
