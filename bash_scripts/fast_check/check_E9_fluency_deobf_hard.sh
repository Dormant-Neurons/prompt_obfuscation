#!/bin/bash

# This script runs a FAST, MINIMAL configuration of the Experiment E9 pipeline.
# Its purpose is to perform a functional check of the code within minutes,
# not to reproduce the paper's results. The output scores will be poor.

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

echo -e "${BLUE}--- Running FAST CHECK for E9: Hard Fluency Deobfuscation ---${NC}"
echo -e "${BLUE}--- Using obfuscation results from: $RESULTS_DIR ---${NC}"

# --- Step 1: Run the hard deobfuscation attack ---
echo -e "\n${CYAN}[1/3] Running hard fluency deobfuscation...${NC}"
python3 fluency_deobfuscation.py \
    --results_dir "$RESULTS_DIR" \
    --embedding_file "$RESULTS_DIR/best_candidate.pt" \
    --deobfuscation_method hard \
    --deobfuscated_sys_prompts_filename "deobfuscated_sys_prompt_list_hard.pt" \
    --optimizer_iter 2 \
    --output_token_count 5 \
    --dataset_size 100

# --- Step 2: Evaluate the best deobfuscated prompt ---
echo -e "\n${CYAN}[2/3] Evaluating the result of the hard deobfuscation...${NC}"
python3 evaluate_fluency_deobfuscation.py \
    --results_dir "$RESULTS_DIR" \
    --sys_prompt_list_file "$RESULTS_DIR/deobfuscated_sys_prompt_list_hard.pt" \
    --best_candidate_filename "best_deobf_sys_prompt_hard.pt" \
    --best_candidate_scores_filename "best_deobf_sys_prompt_hard_scores.json"

# --- Step 3: Compare conventional prompt vs. random baseline ---
echo -e "\n${CYAN}[3/3] Comparing conventional prompt vs. random baseline...${NC}"
python3 compare_sys_prompts.py \
    --results_dir "$RESULTS_DIR" \
    --sys_prompt_1_conventional \
    --sys_prompt_2_random \
    --output_dir "$RESULTS_DIR" \
    --scores_filename "random_sys_prompt_scores.json" \
    --seed 43

echo -e "\n${BLUE}---${NC}"
echo -e "${GREEN}✅ FAST CHECK for E9 completed successfully!${NC}"
echo "You can now find the result files in the '$RESULTS_DIR' directory."
echo "  - Deobfuscated Scores: '$RESULTS_DIR/best_deobf_sys_prompt_hard_scores.json' (for 'deobf' column)"
echo "  - Random Baseline Scores: '$RESULTS_DIR/random_sys_prompt_scores.json' (for 'rand' column)"