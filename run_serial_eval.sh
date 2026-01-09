#!/bin/bash

# ================= Configuration Area =================

# 1. Parameter check and parsing
# Requirement: At least 4 arguments (GPU_ID, MODEL_NAME, RATIOS_STRING, METHOD_1)
if [ $# -lt 4 ]; then
    echo "❌ Error: Insufficient parameters."
    echo "Usage: $0 <gpu_id> <model_name> \"<ratio_list>\" <method_name_1> [method_name_2 ...]"
    echo "Note: ratio_list must be enclosed in quotes, with multiple ratios separated by spaces."
    echo "Example: $0 0 \"Qwen2.5-VL-7B-Instruct\" \"0.75 0.9\" idpruner_lambda0.5"
    exit 1
fi

# Get GPU ID (1st argument)
GPU_ID=$1

# Get Model Name (2nd argument)
MODEL=$2

# Get Pruning Ratio list (3rd argument, passed as a string)
RATIOS_INPUT=$3

# Convert the input string into an array (using space as delimiter)
IFS=' ' read -r -a RATIOS_LIST <<< "$RATIOS_INPUT"

# Remove the first three arguments ($GPU_ID, $MODEL, $RATIOS_INPUT), the rest are methods
shift 3

# Get the list of methods (remaining arguments)
METHODS_LIST=("$@")

# 3. Task List
TASKS=("textvqa" "mme" "pope" "docvqa" "gqa" "scienceqa_img" "ocrbench" "vizwiz_vqa" "mmstar" "chartqa" "ai2d" "mmbench_en_dev" "mmbench_cn_dev")

# =====================================================

# Define interrupt handler function
on_interrupt() {
    echo ""
    echo "🛑 Interrupt signal received (Ctrl+C). Terminating all tasks and exiting script..."
    exit 130
}

# Register Trap: Catch SIGINT (Ctrl+C) and SIGTERM
trap on_interrupt SIGINT SIGTERM

echo "🚀 Starting serial evaluation script (Target GPU: $GPU_ID)..."
echo "Model: $MODEL"
echo "Methods: ${METHODS_LIST[*]}"
echo "Ratios: ${RATIOS_LIST[*]}"
echo "--------------------------------"

# 1. Iterate through methods
for method in "${METHODS_LIST[@]}"; do
    echo ""
    echo "📦 [Method] Starting evaluation for: $method"
    
    # 2. Iterate through pruning ratios
    for ratio in "${RATIOS_LIST[@]}"; do
        
        echo "    Arguments: Ratio=$ratio | GPU=$GPU_ID"
        
        # 3. Iterate through tasks
        for task in "${TASKS[@]}"; do
            echo "    ▶️  Executing: Task=$task (Ratio=$ratio, GPU=$GPU_ID)..."
            
            # Execute the command in the foreground
            CUDA_VISIBLE_DEVICES=$GPU_ID python -m run.run_pruned_method_eval \
                --models "$MODEL" \
                --methods "$method" \
                --tasks "$task" \
                --ratios "$ratio" \
                --layers 0
            
            # Get the exit code of the last command
            exit_code=$?
            
            # Check the exit code
            if [ $exit_code -eq 0 ]; then
                echo "    ✅ Task $task completed."
            elif [ $exit_code -eq 130 ]; then
                # 130 is the standard exit code for a process terminated by SIGINT
                echo ""
                echo "🛑 Detected Python process interrupted by user. Exiting main script."
                exit 130
            else
                # For other error codes (e.g., 1), print warning and continue to next task
                echo "⚠️  Task $task failed (Exit Code: $exit_code). Skipping and proceeding to next task..."
            fi
            
        done
        echo "    ----- Ratio $ratio Finished -----"
    done
    echo "✅ Method $method evaluation completed."
done

echo "🎉 All scheduled tasks finished successfully!"