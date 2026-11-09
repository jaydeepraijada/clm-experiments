#!/usr/bin/env bash
# Run one pipeline stage inside the right NVIDIA container on the DGX Spark.
#
#   ./run.sh stage_data --dry-run
#   ./run.sh train_heads --stage pre --max-steps 30000
#   ./run.sh --bg pre train_heads --stage pre --max-steps 30000   # in tmux session "clm-pre", survives SSH drops
#   ./run.sh shell                                                # interactive shell in the training image
#
# DATA_DIR (default ~/clm-data) and HF_HOME (default ~/.cache/huggingface) are mounted at the same paths inside
# the container, so checkpoint paths are identical on the host and in the container. Logs: $DATA_DIR/logs/.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_DIR="${DATA_DIR:-$HOME/clm-data}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
TRAIN_IMAGE="${TRAIN_IMAGE:-nvcr.io/nvidia/pytorch:26.06-py3}"
VLLM_IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.05.post1-py3}"

if [[ "${1:-}" == "--bg" ]]; then
    name="$2"; shift 2
    mkdir -p "$DATA_DIR/logs"
    log="$DATA_DIR/logs/$name-$(date +%Y%m%d-%H%M%S).log"
    tmux new-session -d -s "clm-$name" "cd '$REPO' && ./run.sh $(printf '%q ' "$@") 2>&1 | tee -a '$log'"
    echo "started tmux session clm-$name, log $log"
    echo "  follow: tail -f $log    attach: tmux attach -t clm-$name"
    exit 0
fi

stage="${1:?usage: ./run.sh [--bg NAME] <stage|shell> [args...]}"; shift
case "$stage" in
    stage_data|train_heads|shell) image="$TRAIN_IMAGE" ;;
    *) echo "unknown or not-yet-ported stage: $stage" >&2; exit 2 ;;
esac

mkdir -p "$DATA_DIR/.home"
args=(--rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
      --user "$(id -u):$(id -g)" -e HOME="$DATA_DIR/.home" -e USER="$(id -un)"
      -e DATA_DIR -e HF_HOME -e PYTHONUNBUFFERED=1
      -v "$REPO:$REPO" -v "$DATA_DIR:$DATA_DIR" -v "$HF_HOME:$HF_HOME" -w "$REPO")

if [[ "$stage" == "shell" ]]; then
    exec docker run -it "${args[@]}" --name "clm-shell-$$" "$image" bash
fi
exec docker run "${args[@]}" --name "clm-$stage-$$" "$image" python pipeline/"$stage".py "$@"
