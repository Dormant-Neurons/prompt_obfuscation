#!/bin/bash

# This script runs the LoRA fine-tuning and evaluation pipeline to reproduce
# a single data point for the "Task" scenario in Table 6 of the paper.

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
OUTPUT_DIR="results/finetuning_${DATASET_NAME}_task_${RUN_ID}"

echo -e "${BLUE}--- Running LoRA Fine-tuning (Task Scenario) on dataset: $DATASET_NAME ---${NC}"
echo -e "${BLUE}--- Output directory: $OUTPUT_DIR ---${NC}"


# --- Step 1: Run the LoRA fine-tuning process (with blank style prompt) ---
echo -e "\n${CYAN}[1/2] Running LoRA fine-tuning...${NC}"
python3 finetune.py \
    --system_prompt "" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    # Reduce these parameters for faster computation (but potentially worse results)
    #--optimizer_iter 10 \
    #--output_token_count 15 \
    #--dataset_size 800 \
    # Increase this parameter for faster computation (but higher VRAM)
    #--batch_size 4

# --- Step 2: Evaluate the best fine-tuned adapter ---
echo -e "\n${CYAN}[2/2] Evaluating the fine-tuned adapters...${NC}"
python3 evaluate_finetuning.py \
    --results_dir "$OUTPUT_DIR" \
    # Increase this parameter for faster computation (but higher VRAM)
    #--eval_batch_size 32

echo -e "\n${BLUE}---${NC}"
echo -e "${GREEN}✅ Successfully completed fine-tuning and evaluation for dataset '$DATASET_NAME' (Task Scenario)${NC}"
echo "You can now find the result file in the '$OUTPUT_DIR' directory to verify the results in Table 6."
echo "  - Fine-tuning Scores: '$OUTPUT_DIR/best_adapter_scores.json' (for 'finetune' column)"
echo "Compare these scores with the 'obf' scores from the corresponding soft prompt obfuscation run."