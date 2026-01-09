#!/bin/bash

# ================= 配置区域 =================

# MODEL="Qwen2.5-VL-7B-Instruct"  <-- 已移除硬编码，改为下方参数获取

# 1. 参数检查与解析
# 修改：需要至少4个参数 (GPU_ID, MODEL_NAME, RATIOS_STRING, METHOD_1)
if [ $# -lt 4 ]; then
    echo "❌ 错误: 参数不足。"
    echo "用法: $0 <gpu_id> <model_name> \"<ratio_list>\" <method_name_1> [method_name_2 ...]"
    echo "注意: ratio_list 必须用引号包围，多个比例用空格分隔。"
    echo "示例: $0 0 \"Qwen2.5-VL-7B-Instruct\" \"0.75 0.9\" cd_pruning_vision_selector_non_init_softmax"
    exit 1
fi

# 获取 GPU ID (第一个参数)
GPU_ID=$1

# 获取 模型名称 (第二个参数) [New]
MODEL=$2

# 获取 剪枝率列表 (第三个参数，作为字符串传入) [Cite: run_serial_eval.sh]
RATIOS_INPUT=$3

# 将输入的字符串转换为数组 (以空格为分隔符)
IFS=' ' read -r -a RATIOS_LIST <<< "$RATIOS_INPUT"

# 移除前三个参数 (GPU_ID, MODEL, RATIOS_INPUT)，剩下的就是方法列表
shift 3

# 获取方法列表 (剩余参数)
METHODS_LIST=("$@")

# 3. 任务列表 (保持不变) [Cite: run_serial_eval.sh]
TASKS=("textvqa" "mme" "pope" "docvqa" "gqa" "scienceqa_img" "ocrbench" "vizwiz_vqa" "mmstar" "chartqa" "ai2d" "mmbench_en_dev" "mmbench_cn_dev")

# ===========================================

# 定义中断处理函数
on_interrupt() {
    echo ""
    echo "🛑 接收到中断信号 (Ctrl+C)，正在终止所有任务并退出脚本..."
    exit 130
}

# 注册 Trap：捕获 SIGINT (Ctrl+C) 和 SIGTERM
trap on_interrupt SIGINT SIGTERM

echo "🚀 开始串行评测脚本 (指定 GPU: $GPU_ID)..."
echo "模型: $MODEL"
echo "方法列表: ${METHODS_LIST[*]}"
echo "剪枝率列表: ${RATIOS_LIST[*]}"
echo "--------------------------------"

# 1. 遍历方法
for method in "${METHODS_LIST[@]}"; do
    echo ""
    echo "📦 [Method] 开始评测方法: $method"
    
    # 2. 遍历剪枝率
    for ratio in "${RATIOS_LIST[@]}"; do
        
        echo "    Arguments: Ratio=$ratio | GPU=$GPU_ID"
        
        # 3. 遍历任务
        for task in "${TASKS[@]}"; do
            echo "    ▶️  正在执行: Task=$task (Ratio=$ratio, GPU=$GPU_ID)..."
            
            # 直接在前台执行命令
            CUDA_VISIBLE_DEVICES=$GPU_ID python -m run.run_pruned_method_eval \
                --models "$MODEL" \
                --methods "$method" \
                --tasks "$task" \
                --ratios "$ratio" \
                --layers 0
            
            # 获取上一个命令的退出码
            exit_code=$?
            
            # 检查退出码
            if [ $exit_code -eq 0 ]; then
                echo "    ✅ Task $task 完成。"
            elif [ $exit_code -eq 130 ]; then
                # 130 是 Bash 中进程被 SIGINT (Ctrl+C) 终止的标准退出码
                echo ""
                echo "🛑 检测到 Python 进程被用户中断，退出主脚本。"
                exit 130
            else
                # 其他错误码 (如 1)，仅打印警告，继续下一个任务
                echo "⚠️  Task $task 执行失败 (Exit Code: $exit_code)。正在跳过并继续下一个任务..."
            fi
            
        done
        echo "    ----- Ratio $ratio 完成 -----"
    done
    echo "✅ 方法 $method 全部完成。"
done

echo "🎉 所有计划任务全部完成！"