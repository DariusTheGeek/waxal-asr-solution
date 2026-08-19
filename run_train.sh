#!/bin/bash
set -e

# Train launcher.
#
# Usage:
#   bash run_train.sh [MODEL] [LANG]
#
# MODEL:
#   ctc3b | ctc1b | llm3b | llm1b | ctc7b | mms1b | joint | all
#   default: all
#
# LANG:
#   lin | sna | all
#   default: all
#
# Examples:
#   bash run_train.sh
#     -> train every model for both languages
#
#   bash run_train.sh llm3b sna
#     -> train only the Shona OmniASR LLM-3B
#
#   bash run_train.sh mms1b all
#     -> train MMS-1B for both languages
#
#   bash run_train.sh joint
#     -> train the tag-free bilingual routing model (language is ignored)
#
# Each model reads its hyperparameters from configs/<lang>/<model>.yaml.
# OmniASR models train under .venvs/omni; MMS models under .venvs/hf.
#
# Full training is multi-GPU and takes days. To check that the code runs
# without committing to that, add --smoke:
#   bash run_train.sh llm1b lin --smoke

DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_SEL=${1:-all}
LANG_SEL=${2:-all}
shift 2 2>/dev/null || true
EXTRA=("$@")

VALID_MODELS="ctc3b ctc1b llm3b llm1b ctc7b mms1b joint all"
VALID_LANGS="lin sna all"

if [[ ! " $VALID_MODELS " =~ " $MODEL_SEL " ]]; then
    echo "Usage: bash run_train.sh [ctc3b|ctc1b|llm3b|llm1b|ctc7b|mms1b|joint|all] [lin|sna|all]"
    exit 1
fi
if [[ ! " $VALID_LANGS " =~ " $LANG_SEL " ]]; then
    echo "Usage: bash run_train.sh [ctc3b|ctc1b|llm3b|llm1b|ctc7b|mms1b|joint|all] [lin|sna|all]"
    exit 1
fi

if [[ "$MODEL_SEL" == "all" ]]; then MODELS=(ctc3b ctc1b llm3b llm1b ctc7b mms1b joint); else MODELS=("$MODEL_SEL"); fi
if [[ "$LANG_SEL"  == "all" ]]; then LANGS=(lin sna);                             else LANGS=("$LANG_SEL");   fi

for M in "${MODELS[@]}"; do
    # The joint routing model is bilingual by construction; it has no lane.
    if [[ "$M" == "joint" ]]; then LOOP=(joint); else LOOP=("${LANGS[@]}"); fi
    for L in "${LOOP[@]}"; do
        CFG="$DIR/configs/$L/$M.yaml"
        if [[ ! -f "$CFG" ]]; then
            echo ">>> Skipping $M / $L (no config: configs/$L/$M.yaml)"
            continue
        fi
        if [[ "$M" == "mms1b" ]]; then
            PY="$DIR/.venvs/hf/bin/python";   SCRIPT="$DIR/train/train_mms.py"
        else
            PY="$DIR/.venvs/omni/bin/python"; SCRIPT="$DIR/train/train_omniasr.py"
        fi
        if [[ ! -x "$PY" ]]; then
            echo "Missing environment: $PY (run: bash install.sh)"; exit 1
        fi
        echo ">>> Training $M / $L"
        "$PY" "$SCRIPT" --config "$CFG" "${EXTRA[@]}"
    done
done

echo ">>> Training complete: models=${MODEL_SEL}, langs=${LANG_SEL}"
