# Controlled DistilBERT input-length experiment: max_length 128 vs 256 vs 512.
#
#   .\scripts\run_maxlen_experiment.ps1 -Labels data\labels\llama31_8b_q4km_final\labels_final.jsonl
#   .\scripts\run_maxlen_experiment.ps1 -Labels <file> -Device cpu      # while the GPU is busy
#
# ONLY max_length varies. Data file, seed, split, optimizer, architecture, loss, epochs and
# batch size are held fixed, so a difference in test MAE is attributable to prompt context length
# and nothing else. No hyperparameter search happens here.
#
# Motivation: max_length 128 was chosen against the synthetic file. The real LMSYS subset has a
# p90 prompt around 380 tokens, so 128 truncates a large share of real prompts before DistilBERT
# ever sees them. 512 is expected to win, but that is a hypothesis until measured.

param(
    [Parameter(Mandatory = $true)][string]$Labels,
    [int[]] $MaxLengths = @(128, 256, 512),
    [string]$Device = "auto",
    [int]   $Epochs = 3,
    [int]   $Seed = 42,
    [string]$Tag = "maxlen",
    [string]$ArtifactRoot = "artifacts",
    # The baseline needs a Llama token count. llama-server gives the real tokenizer; if it is not
    # running, "distilbert" keeps the comparison runnable (and is noted in the results).
    [string]$PromptTokenCounter = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Labels)) { throw "label file not found: $Labels" }

if (-not $PromptTokenCounter) {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:8080/health" -UseBasicParsing -TimeoutSec 5 | Out-Null
        $PromptTokenCounter = "llamacpp:http://127.0.0.1:8080"
        Write-Host "llama-server reachable - baseline will use the real Llama tokenizer"
    } catch {
        $PromptTokenCounter = "distilbert"
        Write-Warning "llama-server unreachable - baseline falls back to the DistilBERT tokenizer. Prompt-token counts are then not Llama's."
    }
}

if ($Device -eq "auto" -or $Device -eq "cuda") {
    if (Test-Path "logs\generation.pid") {
        $gp = Get-Content "logs\generation.pid"
        if (Get-Process -Id $gp -ErrorAction SilentlyContinue) {
            Write-Warning "base generation (PID $gp) is still running. GPU training would contend with it and the latency numbers would not be final. Re-run with -Device cpu, or wait."
        }
    }
}

$dirs = @()
foreach ($ml in $MaxLengths) {
    $out = Join-Path $ArtifactRoot "${Tag}_${ml}"
    Write-Host "`n=== max_length=$ml -> $out ==="
    & $py -m ml.train --real --data $Labels --output-dir $out `
        --max-length $ml --epochs $Epochs --seed $Seed `
        --prompt-token-counter $PromptTokenCounter
    if ($LASTEXITCODE -ne 0) { throw "training failed at max_length=$ml" }
    $dirs += $out
}

Write-Host "`n=== comparison ==="
& $py -m ml.compare_runs @dirs
