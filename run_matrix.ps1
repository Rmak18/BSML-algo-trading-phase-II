# run_matrix.ps1 — run all policies × seeds that exist locally
$ErrorActionPreference = "Stop"

$policies = @("baseline","uniform_policy","ou","pink")  # will skip missing modules
$seeds    = @(101,202,303)

# set PYTHONPATH
$env:PYTHONPATH = (Resolve-Path ".\src").Path

# helper: check module import
function Test-PolicyModule([string]$pol) {
  $code = "import importlib,sys,os; sys.path.insert(0, os.path.abspath('src')); importlib.import_module('bsml.policies.$pol')"
  python -c $code 2>$null
  if ($LASTEXITCODE -eq 0) { return $true } else { return $false }
}

foreach ($pol in $policies) {
  if (-not (Test-PolicyModule $pol)) {
    Write-Host "Skip policy '$pol' (module not present)" -ForegroundColor Yellow
    continue
  }
  foreach ($seed in $seeds) {
    # update configs/run.yaml in-place
    (Get-Content .\configs\run.yaml) `
      | ForEach-Object { $_ -replace '^(policy:\s*).+$', "`$1$pol" } `
      | ForEach-Object { $_ -replace '^(seed:\s*).+$',   "`$1$seed" } `
      | Set-Content .\configs\run.yaml

    Write-Host ("Running policy={0} seed={1}" -f $pol, $seed) -ForegroundColor Cyan
    python -m bsml.core.runner
  }
}
Write-Host "Done."
