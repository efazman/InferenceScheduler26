#!/usr/bin/env bash
# Emit one line and exit when the three real runs finish, or when they cannot finish.
# Coverage: silence must never be mistaken for progress, so every terminal state emits.
set -u
cd "$(dirname "$0")/.." || exit 1
DRIVER_PID=${DRIVER_PID:-$(cat logs/real_runs.pid 2>/dev/null || echo 0)}
POLL=${POLL:-60}
alive() { tasklist //FI "PID eq $1" //NH 2>/dev/null | grep -q "$1"; }

while true; do
    done_runs=""
    for p in fifo sejf adaptive; do
        [ -f "scheduler_runs/real-$p/summary.json" ] && done_runs="$done_runs $p"
    done
    n=$(echo $done_runs | wc -w)

    if [ "$n" -ge 3 ]; then
        echo "REAL RUNS COMPLETE: fifo, sejf, adaptive all have summary.json. Ready for report + Tiger import."
        exit 0
    fi
    if grep -qiE "Traceback|^error:|failed with exit" logs/real_runs.log logs/real_runs.err.log 2>/dev/null; then
        echo "REAL RUNS ERROR at ${n}/3 complete ($done_runs). Check logs/real_runs.log and logs/real_runs.err.log."
        exit 1
    fi
    if ! alive "$DRIVER_PID"; then
        echo "REAL RUNS DRIVER GONE (PID $DRIVER_PID) with only ${n}/3 complete ($done_runs). Relaunch: .\\scripts\\run_real_experiments.ps1 -Detach"
        exit 1
    fi
    if ! alive 1420; then
        echo "LLAMA-SERVER DIED (PID 1420) at ${n}/3 runs complete. Restart: .\\scripts\\start_llama_server.ps1 -Background"
        exit 1
    fi
    sleep "$POLL"
done
