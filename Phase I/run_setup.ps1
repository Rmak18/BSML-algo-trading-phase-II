# --- run_setup.ps1 (safe version) ---
# Set PYTHONPATH to ./src, verify import, then run the runner.

$ErrorActionPreference = "Stop"

# 1) Set PYTHONPATH
$srcPath = (Resolve-Path ".\src").Path
$env:PYTHONPATH = $srcPath
Write-Host ""
Write-Host "PYTHONPATH set to: $srcPath"

# 2) Quick import check (baseline module must exist)
python -c "import importlib,sys,os; sys.path.insert(0, os.path.abspath('src')); importlib.import_module('bsml.policies.baseline'); print('import ok')"
if ($LASTEXITCODE -ne 0) {
  Write-Host "Import failed. Check that src\bsml\policies\baseline.py exists." -ForegroundColor Red
  exit 1
}

# 3) Run the runner
Write-Host ""
Write-Host "Running backtest..."
python -m bsml.core.runner

# 4) End
Write-Host ""
Write-Host ("Finished at {0}" -f (Get-Date -Format "HH:mm:ss"))

