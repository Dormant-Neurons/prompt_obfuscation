#!/bin/bash

# This script runs the prompt extraction attack pipeline to reproduce the
# results for Table 7 of the paper. It tests both an obfuscated soft prompt
# and the original conventional prompt to demonstrate the effectiveness of the obfuscation.
# It should be run AFTER a soft prompt obfuscation experiment.

# Exit immediately if any command fails
set -e

# --- Color Definitions ---
BLUE='\033[1;34m'
CYAN='\033[0;36m'
GREEN='\033[1;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color


# --- Argument Parsing ---
if [ "$#" -ne 2 ] || [ "$1" != "--results_dir" ]; then
    echo -e "${BLUE}Usage: $0 --results_dir <path_to_soft_obfuscation_results>${NC}"
    echo "  Example: $0 --results_dir results/soft_pirate_truthfulqa_full"
    exit 1
fi

RESULTS_DIR=$2

# --- Prerequisite Check ---
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${RED}Error: Results directory '$RESULTS_DIR' not found. Please run a soft obfuscation experiment first.${NC}"
    exit 1
fi
if [ ! -f "$RESULTS_DIR/best_candidate.pt" ]; then
    echo -e "${RED}Error: Required file '$RESULTS_DIR/best_candidate.pt' not found. Please run the soft obfuscation evaluation script first.${NC}"
    exit 1
fi

echo -e "${BLUE}--- Running Prompt Extraction Attack (for Table 7) ---${NC}"
echo -e "${BLUE}--- Using obfuscation results from: $RESULTS_DIR ---${NC}"

# --- Step 1: Run attack against the OBFUSCATED prompt ---
echo -e "\n${CYAN}[1/4] Running prompt extraction attack on the OBFUSCATED soft prompt...${NC}"
python3 prompt_extraction.py \
    --results_dir "$RESULTS_DIR" \
    --extraction_prompts_file "extraction_prompts/gpt4_generated.json" \
    --output_filename "prompt_extraction_output_obfuscated.json" \
    --tensor_file "$RESULTS_DIR/best_candidate.pt" \
    # Increase this parameter for faster computation (but higher VRAM)
    # --batch_size 32

# --- Step 2: Run attack against the CONVENTIONAL prompt (for baseline) ---
echo -e "\n${CYAN}[2/4] Running prompt extraction attack on the CONVENTIONAL prompt...${NC}"
python3 prompt_extraction.py \
    --results_dir "$RESULTS_DIR" \
    --extraction_prompts_file "extraction_prompts/gpt4_generated.json" \
    --output_filename "prompt_extraction_output_conventional.json" \
    --conventional \
    # Increase this parameter for faster computation (but higher VRAM)
    # --batch_size 32

# --- Step 3: Evaluate the attack on the OBFUSCATED prompt ---
echo -e "\n${CYAN}[3/4] Evaluating attack results for the OBFUSCATED prompt...${NC}"
python3 evaluate_prompt_extraction.py \
    --results_dir "$RESULTS_DIR" \
    --extraction_output_file "$RESULTS_DIR/prompt_extraction_output_obfuscated.json" \
    --rouge_recall_threshold 0.9 \
    --successful_outputs_filename "successful_outputs_obfuscated.json"

# --- Step 4: Evaluate the attack on the CONVENTIONAL prompt ---
echo -e "\n${CYAN}[4/4] Evaluating attack results for the CONVENTIONAL prompt...${NC}"
python3 evaluate_prompt_extraction.py \
    --results_dir "$RESULTS_DIR" \
    --extraction_output_file "$RESULTS_DIR/prompt_extraction_output_conventional.json" \
    --rouge_recall_threshold 0.9 \
    --successful_outputs_filename "successful_outputs_conventional.json"

echo -e "\n${BLUE}---${NC}"
echo -e "${GREEN}✅ Successfully completed prompt extraction attacks and evaluations.${NC}"
echo "You can now find the result files in the '$RESULTS_DIR' directory to verify the results in Table 7."
echo "The console output from the evaluation scripts shows the number of successful extractions."
echo "  - Obfuscated successful outputs: '$RESULTS_DIR/successful_outputs_obfuscated.json'"
echo "  - Conventional successful outputs: '$RESULTS_DIR/successful_outputs_conventional.json'"