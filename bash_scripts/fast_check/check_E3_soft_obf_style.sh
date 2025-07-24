      
#!/bin/bash

# This script runs a FAST, MINIMAL configuration of the Experiment E3 pipeline.
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
if [ "$#" -ne 4 ] || { [ "$1" != "--style" ] && [ "$1" != "--dataset_name" ]; } || { [ "$3" != "--style" ] && [ "$3" != "--dataset_name" ]; }; then
    echo -e "${BLUE}Usage: $0 --style <style_name> --dataset_name <dataset>${NC}"
    echo "  <style_name>: The style to use (e.g., pirate, poem). Defined in src/style_prompts.py."
    echo "  <dataset>: The dataset to use (truthfulqa, triviaqa, cnn_dailymail, samsum)."
    exit 1
fi

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --style) STYLE="$2"; shift ;;
        --dataset_name) DATASET_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

RUN_ID=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="results/fast_check/E3_soft_${STYLE}_${DATASET_NAME}_style_${RUN_ID}"

echo -e "${BLUE}--- Running FAST CHECK for E3: Soft Prompt Obfuscation (Style Scenario) ---${NC}"
echo -e "${BLUE}--- Using style: '$STYLE' on dataset: '$DATASET_NAME' ---${NC}"
echo -e "${BLUE}--- Output directory: $OUTPUT_DIR ---${NC}"
echo -e "${BLUE}NOTE: This is a functional test. Final scores are expected to be low.${NC}"

# --- Step 1: Run Obfuscation (with task hints enabled) ---
echo -e "\n${CYAN}[1/6] Running soft prompt obfuscation...${NC}"
python3 obfuscate.py \
    --style "$STYLE" \
    --dataset_name "$DATASET_NAME" \
    --obfuscation_method soft \
    --task_hints \
    --output_dir "$OUTPUT_DIR" \
    --optimizer_iter 2 \
    --output_token_count 5 \
    --dataset_size 100

# --- Step 2: Evaluate Obfuscation to get 'obf' scores ---
echo -e "\n${CYAN}[2/6] Evaluating obfuscated prompt utility...${NC}"
python3 evaluate_obfuscation.py \
    --results_dir "$OUTPUT_DIR" \
    # Increase this parameter for faster computation (but higher VRAM)
    #--eval_batch_size 32

# --- Step 3: Generate 'blank' baseline output ---
echo -e "\n${CYAN}[3/6] Generating baseline output (blank system prompt)...${NC}"
python3 generate_output.py \
    --results_dir "$OUTPUT_DIR" \
    --dataset_file "$OUTPUT_DIR/prepared_data/test_data.json" \
    --output_filename "blank_sys_prompt_output.json" \
    --blank \
    # Increase this parameter for faster computation (but higher VRAM)
    # --batch_size 32

# --- Step 4: Generate 'original' baseline output (different seed) ---
echo -e "\n${CYAN}[4/6] Generating baseline output (conventional prompt, different seed)...${NC}"
python3 generate_output.py \
    --results_dir "$OUTPUT_DIR" \
    --dataset_file "$OUTPUT_DIR/prepared_data/test_data.json" \
    --output_filename "conventional_sys_prompt_seed_43_output.json" \
    --conventional \
    --seed 43 \
    # Increase this parameter for faster computation (but higher VRAM)
    # --batch_size 32

# --- Step 5: Compare conventional vs. blank to get 'blank' scores ---
echo -e "\n${CYAN}[5/6] Comparing conventional vs. blank baseline...${NC}"
python3 compare_output.py \
    --output_file_1 "$OUTPUT_DIR/conventional_output.json" \
    --output_file_2 "$OUTPUT_DIR/blank_sys_prompt_output.json" \
    --output_dir "$OUTPUT_DIR" \
    --scores_filename "blank_output_scores.json"

# --- Step 6: Compare conventional vs. conventional to get 'original' scores ---
echo -e "\n${CYAN}[6/6] Comparing conventional vs. conventional (different seed) baseline...${NC}"
python3 compare_output.py \
    --output_file_1 "$OUTPUT_DIR/conventional_output.json" \
    --output_file_2 "$OUTPUT_DIR/conventional_sys_prompt_seed_43_output.json" \
    --output_dir "$OUTPUT_DIR" \
    --scores_filename "original_output_scores.json"

echo -e "\n${BLUE}---${NC}"
echo -e "${GREEN}✅ FAST CHECK for E3 and style '$STYLE' on dataset '$DATASET_NAME' completed successfully! (Style Scenario)${NC}"
echo "You can now find all result files in the '$OUTPUT_DIR' directory."
echo "  - Obfuscated Prompt Scores: '$OUTPUT_DIR/best_candidate_scores.json' (for 'obf' column)"
echo "  - Blank Baseline Scores:    '$OUTPUT_DIR/blank_output_scores.json' (for 'blank' column)"
echo "  - Original Baseline Scores: '$OUTPUT_DIR/original_output_scores.json' (for 'original' column)"

    