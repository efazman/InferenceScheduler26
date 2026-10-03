# Start llama-server for label generation. Pinned so the backend config cannot drift between runs.
#
#   .\scripts\start_llama_server.ps1              # foreground
#   .\scripts\start_llama_server.ps1 -Background   # detached, logs to logs\llama-server.log
#
# Why these flags:
#   -ngl 99     offload every layer to the RTX 3060 Ti (Q4_K_M 8B ~4.6 GiB + 4096-token KV ~0.5 GiB
#               fits in 8 GB). Verify "load_tensors: offloaded 33/33 layers to GPU" in the log.
#   -np 1       ONE slot. The server cannot batch or reorder concurrent requests, so our client's
#               request order is the execution order. This is the measurement-validity constraint
#               for the scheduler experiments; do not raise it for label generation.
#   -c 4096     context per slot. Prompts are capped at 8000 chars (~2000 tokens) upstream, leaving
#               room for the 512-token completion.
#   --no-webui  no browser UI; this server exists only for the client.
#
# temperature / top_p / max_tokens / system prompt are NOT set here. The client sends them per
# request from datagen/config.py, so the recorded generation_config.json is the single source of
# truth for how labels were produced.

param(
    [string]$Model   = "models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    [string]$BinDir  = "vendor\llama.cpp-b11381-cuda-12.4",
    [int]   $Port    = 8080,
    [string]$HostAddr = "127.0.0.1",
    [int]   $Ctx     = 4096,
    [int]   $NGL     = 99,
    [int]   $Slots   = 1,
    [switch]$Background
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$exe = Join-Path $BinDir "llama-server.exe"
if (-not (Test-Path $exe))   { throw "llama-server.exe not found at $exe" }
if (-not (Test-Path $Model)) { throw "model not found at $Model" }

$modelInfo = Get-Item $Model
Write-Host "llama-server : $exe"
Write-Host "model        : $Model ($([math]::Round($modelInfo.Length/1GB,2)) GiB)"
Write-Host "endpoint     : http://${HostAddr}:${Port}  (slots=$Slots, ctx=$Ctx, ngl=$NGL)"

$serverArgs = @(
    "-m", $Model,
    "-c", $Ctx,
    "-ngl", $NGL,
    "-np", $Slots,
    "--host", $HostAddr,
    "--port", $Port,
    "--no-webui"
)

if ($Background) {
    New-Item -ItemType Directory -Force logs | Out-Null
    $log = "logs\llama-server.log"
    $proc = Start-Process -FilePath $exe -ArgumentList $serverArgs -NoNewWindow -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "logs\llama-server.err.log"
    $proc.Id | Out-File -Encoding ascii "logs\llama-server.pid"
    Write-Host "started PID $($proc.Id); log: $log"
} else {
    & $exe @serverArgs
}
