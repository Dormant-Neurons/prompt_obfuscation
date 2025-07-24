#!/bin/bash

# This script runs the token space projection attack to reproduce the
# results for Table 8 of the paper. It projects an obfuscated soft prompt
# back to tokens using two different distance metrics and compares their
# similarity to the original prompt against a random baseline.

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

echo -e "${BLUE}--- Running Token Space Projection Attack (for Table 8) ---${NC}"
echo -e "${BLUE}--- Using obfuscation results from: $RESULTS_DIR ---${NC}"

# --- Step 1: Project soft prompt using Euclidean distance ---
echo -e "\n${CYAN}[1/5] Projecting soft prompt to token space via Euclidean distance...${NC}"
python3 projection.py \
    --results_dir "$RESULTS_DIR" \
    --embedding_file "$RESULTS_DIR/best_candidate.pt" \
    --euclidean \
    --projected_ids_filename "best_candidate_euclidean_projection.pt"

# --- Step 2: Project soft prompt using Cosine distance ---
echo -e "\n${CYAN}[2/5] Projecting soft prompt to token space via Cosine similarity...${NC}"
python3 projection.py \
    --results_dir "$RESULTS_DIR" \
    --embedding_file "$RESULTS_DIR/best_candidate.pt" \
    --cosine \
    --projected_ids_filename "best_candidate_cosine_projection.pt"

# --- Step 3: Compare conventional prompt vs. Euclidean projection ---
echo -e "\n${CYAN}[3/5] Comparing conventional prompt vs. Euclidean projection...${NC}"
python3 compare_sys_prompts.py \
    --results_dir "$RESULTS_DIR" \
    --sys_prompt_1_conventional \
    --sys_prompt_2_file "$RESULTS_DIR/best_candidate_euclidean_projection.pt" \
    --output_dir "$RESULTS_DIR" \
    --scores_filename "euclidean_sys_prompt_scores.json"


# --- Step 4: Compare conventional prompt vs. Cosine projection ---
echo -e "\n${CYAN}[4/5] Comparing conventional prompt vs. Cosine projection...${NC}"
python3 compare_sys_prompts.py \
    --results_dir "$RESULTS_DIR" \
    --sys_prompt_1_conventional \
    --sys_prompt_2_file "$RESULTS_DIR/best_candidate_cosine_projection.pt" \
    --output_dir "$RESULTS_DIR" \
    --scores_filename "cosine_sys_prompt_scores.json"

# --- Step 5: Compare conventional prompt vs. random baseline ---
echo -e "\n${CYAN}[5/5] Comparing conventional prompt vs. random baseline...${NC}"
python3 compare_sys_prompts.py \
    --results_dir "$RESULTS_DIR" \
    --sys_prompt_1_conventional \
    --sys_prompt_2_random \
    --output_dir "$RESULTS_DIR" \
    --scores_filename "random_sys_prompt_scores.json" \
    --seed 43

echo -e "\n${BLUE}---${NC}"
echo -e "${GREEN}✅ Successfully completed projection attack and evaluations.${NC}"
echo "You can now find the result files in the '$RESULTS_DIR' directory to verify the results in Table 8."
echo "  - Euclidean Projection Scores: '$RESULTS_DIR/euclidean_sys_prompt_scores.json'"
echo "  - Cosine Projection Scores:    '$RESULTS_DIR/cosine_sys_prompt_scores.json'"
echo "  - Random Baseline Scores:      '$RESULTS_DIR/random_sys_prompt_scores.json'"