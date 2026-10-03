# Unattended driver for the 8000-generation label run.
#
#   .\scripts\run_generation.ps1                 # run in this window
#   .\scripts\run_generation.ps1 -Detach         # survive this shell closing; logs to logs\generation.log
#
# What it adds on top of `python -m datagen.generate_labels`:
#   * restarts the generator if it exits non-zero (a dead server, a transport blip). Resuming is
#     the generator's documented behaviour: completed prompts are skipped and completed
#     generations are reused, so a restart never redoes finished work and never starts from zero.
#   * cuts the 250 / 500 / 1000 / 2000 checkpoints as soon as each becomes available, so an
#     overnight run produces its training sets without anyone watching.
#   * stops immediately on a configuration mismatch or a bad subset, which are operator errors a
#     retry would only repeat.
#
# Safe to run while an instance is already running? No - run one at a time. Two writers would
# interleave appends to the same JSONL files.

param(
    [int]   $MaxRestarts = 50,
    [int]   $RestartDelaySeconds = 20,
    [string]$OutputDir = "data\labels\llama31_8b_q4km",
    [string]$Subset = "data\lmsys\subset_2000.jsonl",
    [switch]$Detach
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
New-Item -ItemType Directory -Force logs | Out-Null

if ($Detach) {
    # Relaunch this same script detached, so it outlives the shell that started it.
    $self = Join-Path $PSScriptRoot "run_generation.ps1"
    $a = @("-NoProfile","-ExecutionPolicy","Bypass","-File",$self,
           "-MaxRestarts",$MaxRestarts,"-RestartDelaySeconds",$RestartDelaySeconds,
           "-OutputDir",$OutputDir,"-Subset",$Subset)
    $p = Start-Process -FilePath "powershell.exe" -ArgumentList $a -WindowStyle Hidden -PassThru `
         -RedirectStandardOutput "logs\generation.log" -RedirectStandardError "logs\generation.err.log"
    $p.Id | Out-File -Encoding ascii "logs\generation.pid"
    Write-Host "detached PID $($p.Id)"
    Write-Host "log        logs\generation.log"
    Write-Host "status     .\.venv\Scripts\python.exe -m datagen.status"
    exit 0
}

# Unbuffered, so logs\generation.log stays live for an unattended run. (datagen.status does not
# depend on this - it reads the fsynced JSONL files directly.)
$env:PYTHONUNBUFFERED = "1"
$py = Join-Path $root ".venv\Scripts\python.exe"
$attempt = 0

while ($true) {
    $attempt++
    Write-Host "=== generator attempt $attempt  $(Get-Date -Format s) ==="

    & $py -m datagen.generate_labels --backend llamacpp --output-dir $OutputDir --input $Subset
    $code = $LASTEXITCODE

    # Cut whatever checkpoints are now available, whether or not the generator exited cleanly.
    & $py -m datagen.make_checkpoint --labels (Join-Path $OutputDir "labels.jsonl") --subset $Subset

    if ($code -eq 0) {
        Write-Host "=== generator finished cleanly  $(Get-Date -Format s) ==="
        break
    }

    # A ConfigMismatch / bad-subset exit is an operator error: generate_labels prints "error:" and
    # exits 1 without having done work. Retrying cannot fix it, so surface it instead of looping.
    $tail = Get-Content "logs\generation.log" -Tail 25 -ErrorAction SilentlyContinue
    if ($tail -match "was produced with different settings|duplicate prompt_id|refusing to write MOCK") {
        Write-Error "configuration problem - not retrying. Fix it, then rerun."
        break
    }

    if ($attempt -ge $MaxRestarts) {
        Write-Error "giving up after $attempt attempts (exit $code)."
        break
    }
    Write-Host "exit $code - restarting in ${RestartDelaySeconds}s (resume skips completed work)"
    Start-Sleep -Seconds $RestartDelaySeconds
}

& $py -m datagen.status --output-dir $OutputDir --subset $Subset
