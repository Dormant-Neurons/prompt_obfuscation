#!/bin/bash

# This script runs a FAST, MINIMAL configuration of the Experiment E5 pipeline.
# Its purpose is to perform a functional check of the code within minutes,
# not to reproduce the paper's results. The output scores will be poor.

# Exit immediately if any command fails
set -e

# --- Color Definitions ---
BLUE='\033[1;34m'
CYAN='\033[0;36m'
GREEN='\033[1;32m'
NC='\033[0m' # No Color

# --- Argument Parsing ---
if [ "$#" -ne 2 ] || [ "$1" != "--dataset_name" ]; then
    echo -e "${BLUE}Usage: $0 --dataset_name <dataset>${NC}"
    echo "  <dataset>: The dataset to use (truthfulqa, triviaqa, cnn_dailymail, samsum)."
    exit 1
fi

DATASET_NAME=$2
RUN_ID=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="results/fast_check/E5_finetuning_${DATASET_NAME}_task_${RUN_ID}"

echo -e "${BLUE}--- Running FAST CHECK for E5: LoRa Finetuning (Task Scenario) ---${NC}"
echo -e "${BLUE}--- Dataset: '$DATASET_NAME' ---${NC}"
echo -e "${BLUE}--- Output directory: $OUTPUT_DIR ---${NC}"
echo -e "${BLUE}NOTE: This is a functional test. Final scores are expected to be low.${NC}"


# --- Step 1: Run the LoRA fine-tuning process (with blank style prompt) ---
echo -e "\n${CYAN}[1/2] Running LoRA fine-tuning...${NC}"
python3 finetune.py \
    --system_prompt "" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --optimizer_iter 2 \
    --output_token_count 5 \
    --dataset_size 100

# --- Step 2: Evaluate the best fine-tuned adapter ---
echo -e "\n${CYAN}[2/2] Evaluating the fine-tuned adapters...${NC}"
python3 evaluate_finetuning.py \
    --results_dir "$OUTPUT_DIR" \
    # Increase this parameter for faster computation (but higher VRAM)
    #--eval_batch_size 32

echo -e "\n${BLUE}---${NC}"
echo -e "${GREEN}✅ FAST CHECK for E5 on dataset '$DATASET_NAME' completed successfully! (Task Scenario)${NC}"
echo "You can now find the result file in the '$OUTPUT_DIR' directory."
echo "  - Fine-tuning Scores: '$OUTPUT_DIR/best_adapter_scores.json' (for 'finetune' column)"
echo "Compare these scores with the 'obf' scores from the corresponding soft prompt obfuscation run."