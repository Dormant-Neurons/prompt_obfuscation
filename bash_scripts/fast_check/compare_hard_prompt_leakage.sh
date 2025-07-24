#!/bin/bash

# This script reproduces the prompt similarity results for Table 2,
# measuring the information leakage of a hard obfuscated prompt.
# It should be run AFTER an obfuscation experiment (e.g., hard_prompt_obfuscation_full.sh)
# as it requires the results from that run.

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
    echo -e "${BLUE}Usage: $0 --results_dir <path_to_results_directory>${NC}"
    echo "  Example: $0 --results_dir results/hard_pirate_full"
    exit 1
fi

RESULTS_DIR=$2

# --- Prerequisite Check ---
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${RED}Error: Results directory '$RESULTS_DIR' not found. Please run the corresponding obfuscation experiment first.${NC}"
    exit 1
fi
if [ ! -f "$RESULTS_DIR/best_candidate.pt" ]; then
    echo -e "${RED}Error: Required file '$RESULTS_DIR/best_candidate.pt' not found. Please run the obfuscation evaluation script first.${NC}"
    exit 1
fi

echo -e "${BLUE}--- Measuring Hard Prompt Information Leakage (for Table 2) ---${NC}"
echo -e "${BLUE}--- Using results from: $RESULTS_DIR ---${NC}"

# --- Step 1: Compare conventional prompt to a random prompt (for 'rand' column) ---
echo -e "\n${CYAN}[1/2] Comparing conventional prompt to a random prompt (baseline)...${NC}"
python3 compare_sys_prompts.py \
    --results_dir "$RESULTS_DIR" \
    --sys_prompt_1_conventional \
    --sys_prompt_2_random \
    --output_dir "$RESULTS_DIR" \
    --scores_filename "rand_sys_prompt_scores.json" \
    --seed 43

# --- Step 2: Compare conventional prompt to the best obfuscated prompt (for 'obf' column) ---
echo -e "\n${CYAN}[2/2] Comparing conventional prompt to the best obfuscated prompt...${NC}"
python3 compare_sys_prompts.py \
    --results_dir "$RESULTS_DIR" \
    --sys_prompt_1_conventional \
    --sys_prompt_2_file "$RESULTS_DIR/best_candidate.pt" \
    --output_dir "$RESULTS_DIR" \
    --scores_filename "obf_sys_prompt_scores.json" \
    --seed 43

echo -e "\n${BLUE}---${NC}"
echo -e "${GREEN}✅ Successfully completed prompt similarity comparisons.${NC}"
echo "You can now find the result files in the '$RESULTS_DIR' directory to verify the results in Table 2."
echo "  - Random baseline scores: '$RESULTS_DIR/rand_sys_prompt_scores.json' (for 'rand' column)"
echo "  - Obfuscated scores:      '$RESULTS_DIR/obf_sys_prompt_scores.json' (for 'obf' column)"

    