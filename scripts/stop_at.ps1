# Stop the base label run cleanly once N prompts are complete.
#
#   .\scripts\stop_at.ps1 -Target 500
#   .\scripts\stop_at.ps1 -Target 500 -Detach      # watch in the background
#
# Why a script rather than Ctrl-C at the right moment: the runner
# (scripts/run_generation.ps1) RESTARTS the generator whenever it exits non-zero. Killing the
# python child alone would just make the runner start another one. The runner must go first.
#
# Stopping is safe and reversible:
#   * every completed generation is already fsynced, so at most the one generation in flight is
#     lost, and a half-written final line is repaired on the next start-up;
#   * `labels.jsonl` only ever contains prompts whose 4 runs all finished, so a partially done
#     prompt is simply absent rather than malformed;
#   * resuming later is `.\scripts\run_generation.ps1 -Detach` - completed prompts are skipped and
#     completed generations reused, so nothing is redone and nothing is thrown away.
#
# This does NOT stop llama-server; the extension pass needs it.

param(
    [Parameter(Mandatory = $true)][int]$Target,
    [string]$Labels = "data\labels\llama31_8b_q4km\labels.jsonl",
    [string]$PidFile = "logs\generation.pid",
    [int]   $PollSeconds = 30,
    [switch]$Detach
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if ($Detach) {
    $self = Join-Path $PSScriptRoot "stop_at.ps1"
    $a = @("-NoProfile","-ExecutionPolicy","Bypass","-File",$self,"-Target",$Target,
           "-Labels",$Labels,"-PidFile",$PidFile,"-PollSeconds",$PollSeconds)
    New-Item -ItemType Directory -Force logs | Out-Null
    $p = Start-Process powershell.exe -ArgumentList $a -WindowStyle Hidden -PassThru `
         -RedirectStandardOutput "logs\stop_at.log" -RedirectStandardError "logs\stop_at.err.log"
    $p.Id | Out-File -Encoding ascii "logs\stop_at.pid"
    Write-Host "watcher PID $($p.Id) -> will stop the base run at $Target prompts"
    Write-Host "log: logs\stop_at.log   cancel: Stop-Process -Id $($p.Id)"
    exit 0
}

function Get-CompletedPrompts {
    if (-not (Test-Path $Labels)) { return 0 }
    # The generator holds this file open for append, so it MUST be opened with
    # FileShare::ReadWrite - StreamReader's default share mode throws a sharing violation.
    # Counting only complete lines also means a half-written final line is never miscounted.
    $full = (Resolve-Path $Labels).Path
    $fs = New-Object System.IO.FileStream(
        $full, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite)
    $sr = New-Object System.IO.StreamReader($fs)
    try   {
        $n = 0
        while ($null -ne ($line = $sr.ReadLine())) { if ($line.Trim()) { $n++ } }
        return $n
    }
    finally { $sr.Dispose(); $fs.Dispose() }
}

Write-Host "watching $Labels for $Target completed prompts (poll ${PollSeconds}s)"

while ($true) {
    $n = Get-CompletedPrompts
    if ($n -ge $Target) {
        Write-Host "$(Get-Date -Format s)  reached $n/$Target prompts - stopping the base run"
        break
    }
    if (-not (Test-Path $PidFile)) {
        Write-Warning "$PidFile is gone; the run appears already stopped at $n prompts"
        exit 0
    }
    $runner = Get-Content $PidFile
    if (-not (Get-Process -Id $runner -ErrorAction SilentlyContinue)) {
        Write-Warning "runner PID $runner is no longer alive; stopped on its own at $n prompts"
        exit 0
    }
    Write-Host "$(Get-Date -Format s)  $n/$Target"
    Start-Sleep -Seconds $PollSeconds
}

# Runner first, so it cannot spawn a replacement generator, then its python descendants.
$runner = Get-Content $PidFile
$kids = Get-CimInstance Win32_Process -Filter "ParentProcessId=$runner" |
        Select-Object -ExpandProperty ProcessId
Write-Host "stopping runner $runner"
Stop-Process -Id $runner -Force -ErrorAction SilentlyContinue
foreach ($k in $kids) {
    $grandkids = Get-CimInstance Win32_Process -Filter "ParentProcessId=$k" |
                 Select-Object -ExpandProperty ProcessId
    foreach ($g in $grandkids) { Stop-Process -Id $g -Force -ErrorAction SilentlyContinue }
    Write-Host "stopping child $k"
    Stop-Process -Id $k -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 3

$final = Get-CompletedPrompts
Write-Host "`nbase run stopped at $final completed prompts."
Write-Host "llama-server left running (the extension pass needs it)."
Write-Host "`nnext:"
Write-Host "  .\.venv\Scripts\python.exe -m datagen.extend_censored --backend llamacpp --dry-run"
Write-Host "  .\.venv\Scripts\python.exe -m datagen.extend_censored --backend llamacpp"
Write-Host "  .\.venv\Scripts\python.exe -m datagen.assemble_final_labels"
Write-Host "`nto resume collecting more prompts later:"
Write-Host "  .\scripts\run_generation.ps1 -Detach"
