# Learning curve: train on 500 / 1000 / 2000 prompts with IDENTICAL settings.
#
#   .\scripts\run_learning_curve.ps1 -MaxLength 512
#   .\scripts\run_learning_curve.ps1 -MaxLength 512 -Device cpu
#   .\scripts\run_learning_curve.ps1 -MaxLength 512 -FinalRoot data\labels\llama31_8b_q4km_final\checkpoints
#
# Only the dataset SIZE varies. max_length, seed, split fractions, optimizer, architecture, loss,
# epochs and batch size are fixed across sizes, so the curve measures the value of more data and
# not different training choices. Pass -MaxLength with whatever run_maxlen_experiment.ps1 picked.
#
# Inputs are stable nested checkpoint files - never the still-growing labels.jsonl. Because the
# checkpoints are nested (cp500 is a prefix of cp1000, which is a prefix of cp2000), each larger
# run genuinely adds data rather than resampling it.

param(
    [Parameter(Mandatory = $true)][int]$MaxLength,
    [int[]] $Sizes = @(500, 1000, 2000),
    [string]$Device = "auto",
    [int]   $Epochs = 3,
    [int]   $Seed = 42,
    [string]$Tag = "curve",
    [string]$ArtifactRoot = "artifacts",
    # Default: checkpoints cut from the ASSEMBLED final labels (censoring resolved).
    # Fall back to the base-run checkpoints with -FinalRoot data\labels\llama31_8b_q4km\checkpoints.
    [string]$FinalRoot = "data\labels\llama31_8b_q4km_final\checkpoints",
    [string]$PromptTokenCounter = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"

if (-not $PromptTokenCounter) {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:8080/health" -UseBasicParsing -TimeoutSec 5 | Out-Null
        $PromptTokenCounter = "llamacpp:http://127.0.0.1:8080"
    } catch {
        $PromptTokenCounter = "distilbert"
        Write-Warning "llama-server unreachable - baseline falls back to the DistilBERT tokenizer."
    }
}

$dirs = @()
foreach ($n in $Sizes) {
    $labels = Join-Path $FinalRoot ("checkpoint_{0:d5}\labels_{1}.jsonl" -f $n, $n)
    if (-not (Test-Path $labels)) {
        Write-Warning "skipping $n - $labels does not exist yet"
        continue
    }
    $out = Join-Path $ArtifactRoot "${Tag}_${n}"
    Write-Host "`n=== n=$n (max_length=$MaxLength) -> $out ==="
    & $py -m ml.train --real --data $labels --output-dir $out `
        --max-length $MaxLength --epochs $Epochs --seed $Seed `
        --prompt-token-counter $PromptTokenCounter
    if ($LASTEXITCODE -ne 0) { throw "training failed at n=$n" }
    $dirs += $out
}

if (-not $dirs) { throw "no checkpoints were available; nothing trained" }

Write-Host "`n=== learning curve ==="
& $py -m ml.compare_runs @dirs
