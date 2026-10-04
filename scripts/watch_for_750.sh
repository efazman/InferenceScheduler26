#!/usr/bin/env bash
# Emit exactly one line and exit when the base run reaches its target, or when it becomes
# impossible for it to get there. Used as a Monitor command so the session is woken once.
#
# Coverage matters here: a watch that only greps for success is silent through a crash, and
# silence is indistinguishable from "still running". Every terminal state below emits.
set -u
cd "$(dirname "$0")/.." || exit 1

RUNNER_PID=${RUNNER_PID:-28228}
STOPPER_PID=${STOPPER_PID:-2592}
TARGET=${TARGET:-750}
LABELS=data/labels/llama31_8b_q4km/labels.jsonl
POLL=${POLL:-60}

alive() { tasklist //FI "PID eq $1" //NH 2>/dev/null | grep -q "$1"; }

# Count only newline-terminated lines: the final line may be mid-write.
completed() { [ -f "$LABELS" ] && grep -c '^' "$LABELS" 2>/dev/null || echo 0; }

while true; do
    n=$(completed)

    # 1. success: stop_at.ps1 reports it stopped the run, or the count reached target
    if grep -qE "base run stopped at|stopping the base run" logs/stop_at.log 2>/dev/null; then
        echo "TARGET REACHED: base run stopped at ${n}/${TARGET} prompts. llama-server still up. Ready for the extension pass."
        exit 0
    fi
    if [ "$n" -ge "$TARGET" ]; then
        echo "TARGET REACHED: ${n}/${TARGET} prompts complete (stopper had not logged yet)."
        exit 0
    fi

    # 2. the auto-stop watcher died -> the run will sail past the target unattended
    if ! alive "$STOPPER_PID"; then
        echo "WATCHER GONE: stop_at.ps1 (PID ${STOPPER_PID}) is no longer running at ${n}/${TARGET}. The base run will NOT stop at ${TARGET} on its own."
        exit 1
    fi

    # 3. the generator died -> target is unreachable without a restart
    if ! alive "$RUNNER_PID"; then
        echo "GENERATOR GONE: runner (PID ${RUNNER_PID}) died at ${n}/${TARGET} prompts. Resume with: .\\scripts\\run_generation.ps1 -Detach"
        exit 1
    fi

    # 4. llama-server died -> generations will fail from here
    if ! alive 1420; then
        echo "SERVER GONE: llama-server (PID 1420) died at ${n}/${TARGET}. Restart with: .\\scripts\\start_llama_server.ps1 -Background"
        exit 1
    fi

    sleep "$POLL"
done
