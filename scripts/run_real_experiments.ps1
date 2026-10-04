# Replay ONE frozen manifest under FIFO, SEJF and Adaptive, sequentially.
#
#   .\scripts\run_real_experiments.ps1
#   .\scripts\run_real_experiments.ps1 -Detach
#
# Sequential by design: one GPU and llama-server runs with -np 1 (K=1), so two concurrent runs
# would contend and corrupt the latency measurements. Never run two at once.
#
# Nothing here chooses scheduling order. The manifest fixes the request set, prompts, arrival
# times and per-request seeds; the policy decides execution order. That is the whole experiment.

param(
    [string]$Manifest   = "data\scheduler\real_manifest.json",
    [string]$Predictor  = "distilbert:artifacts\maxlen_512",
    [string]$ServiceFrom = "data\labels\llama31_8b_q4km\runs.jsonl",
    # Comma-separated, NOT [string[]]. `powershell.exe -File` binds only the first
    # space-separated token to an array parameter, so "-Policies fifo sejf adaptive" would
    # silently run fifo only and then report success.
    [string]$Policies = "fifo,sejf,adaptive",
    [string]$Prefix     = "real",
    [switch]$Tiger,
    [switch]$Detach
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force logs | Out-Null

if ($Detach) {
    $self = Join-Path $PSScriptRoot "run_real_experiments.ps1"
    $a = @("-NoProfile","-ExecutionPolicy","Bypass","-File",$self,
           "-Manifest",$Manifest,"-Predictor",$Predictor,"-ServiceFrom",$ServiceFrom,
           "-Policies",$Policies,"-Prefix",$Prefix)
    if ($Tiger) { $a += "-Tiger" }
    $p = Start-Process powershell.exe -ArgumentList $a -WindowStyle Hidden -PassThru `
         -RedirectStandardOutput "logs\real_runs.log" -RedirectStandardError "logs\real_runs.err.log"
    $p.Id | Out-File -Encoding ascii "logs\real_runs.pid"
    Write-Host "detached PID $($p.Id); log: logs\real_runs.log"
    exit 0
}

$policyList = @($Policies -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($policyList.Count -eq 0) { throw "no policies parsed from -Policies '$Policies'" }
Write-Host "policies to run: $($policyList -join ', ')"

$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$py = ".\.venv\Scripts\python.exe"

# Fail before spending GPU time, not halfway through.
if (-not (Test-Path $Manifest)) { throw "manifest not found: $Manifest" }
try { Invoke-WebRequest -Uri "http://127.0.0.1:8080/health" -UseBasicParsing -TimeoutSec 10 | Out-Null }
catch { throw "llama-server is not healthy at 127.0.0.1:8080 - start it with .\scripts\start_llama_server.ps1 -Background" }

foreach ($pol in $policyList) {
    $name = "$Prefix-$pol"
    if (Test-Path "scheduler_runs\$name") {
        Write-Warning "scheduler_runs\$name already exists - skipping (a run never appends; use a new -Prefix to redo)"
        continue
    }
    Write-Host ""
    Write-Host "================ $name  $(Get-Date -Format s) ================"
    $args = @("-m","scheduler","run","--backend","llamacpp","--manifest",$Manifest,
              "--predictor",$Predictor,"--policy",$pol,
              "--median-service-from",$ServiceFrom,"--run-name",$name)
    if ($Tiger) { $args += "--tiger" }
    & $py @args
    if ($LASTEXITCODE -ne 0) { throw "$name failed with exit $LASTEXITCODE" }
}

Write-Host ""
Write-Host "================ all policies done  $(Get-Date -Format s) ================"
Get-ChildItem scheduler_runs -Directory | Where-Object { $_.Name -like "$Prefix-*" } |
    Select-Object Name, LastWriteTime | Format-Table
