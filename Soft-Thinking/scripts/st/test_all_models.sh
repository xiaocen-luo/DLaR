#!/bin/bash
# test_all_models.sh
# 测试三个模型配置在四个数据集上的 pass@8 性能

# 设置基础参数
# /public/home/xiangyuduan/models/hf/DeepSeek-R1-Distill-Qwen-1.5B/
MODEL_PATH="/public/home/xiangyuduan/models/hf/DeepSeek-R1-Distill-Qwen-1.5B/"  # Change the model path
NUM_GPUS=1
NUM_SAMPLES=1  # 1 or 8
MAX_GENERATED_TOKENS=2048
MEM_FRACTION_STATIC=0.3  # 降低以避免 OOM
START_IDX=0
END_IDX=10000

# 创建日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR

# 数据集列表
DATASETS=("gsm8k" "math500" "minervamath" "amc23")
#DATASETS=("gpqa_diamond")

# 模型配置列表
# 格式: "config_name|extra_args"
#CONFIGS=(
#    "Base|"
#    "SoftThinking_Gumbel|--enable_soft_thinking --add_noise_gumbel_softmax --gumbel_softmax_temperature 0.5"
#    "SoftThinking_Dirichlet|--enable_soft_thinking --add_noise_dirichlet --dirichlet_alpha 4.0"
#)

CONFIGS=(
    "Base|"
    "SoftThinking_Gumbel|--enable_soft_thinking --add_noise_gumbel_softmax --gumbel_softmax_temperature 0.5"
)

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================="
echo "开始批量测试"
echo "时间: $(date)"
echo "模型: $MODEL_PATH"
echo "数据集: ${DATASETS[@]}"
echo "配置: ${CONFIGS[@]}"
echo "========================================="

# 遍历数据集
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "========================================="
    echo "开始测试数据集: $DATASET"
    echo "========================================="
    
    # 遍历配置
    for CONFIG in "${CONFIGS[@]}"; do
        # 解析配置名称和参数
        CONFIG_NAME=$(echo $CONFIG | cut -d'|' -f1)
        EXTRA_ARGS=$(echo $CONFIG | cut -d'|' -f2)
        
        # 构建日志文件名
        LOG_FILE="${LOG_DIR}/${DATASET}_${CONFIG_NAME}_${TIMESTAMP}.log"
        
        echo ""
        echo "-----------------------------------------"
        echo "运行配置: $CONFIG_NAME"
        echo "数据集: $DATASET"
        echo "日志: $LOG_FILE"
        echo "额外参数: $EXTRA_ARGS"
        echo "-----------------------------------------"
        
        # 构建命令
        CMD="python run_sglang_softthinking.py \
            --dataset $DATASET \
            --model_name $MODEL_PATH \
            --max_topk 5 \
            --max_generated_tokens $MAX_GENERATED_TOKENS \
            --temperature 0.6 \
            --top_p 0.95 \
            --top_k 30 \
            --min_p 0.0 \
            --after_thinking_temperature 0.6 \
            --after_thinking_top_p 0.95 \
            --after_thinking_top_k 30 \
            --after_thinking_min_p 0.0 \
            --mem_fraction_static $MEM_FRACTION_STATIC \
            --start_idx $START_IDX \
            --end_idx $END_IDX \
            --num_gpus $NUM_GPUS \
            --num_samples $NUM_SAMPLES \
            $EXTRA_ARGS"
        
        echo "执行命令:"
        echo $CMD
        echo ""
        
        # 执行命令并记录日志
        eval $CMD 2>&1 | tee $LOG_FILE
        
        # 检查执行状态
        if [ $? -eq 0 ]; then
            echo "✓ $DATASET - $CONFIG_NAME 完成"
        else
            echo "✗ $DATASET - $CONFIG_NAME 失败"
        fi
        
        # 等待 GPU 内存释放
        echo "等待 GPU 内存释放..."
        sleep 5
        
        # 清理 GPU 缓存（如果 nvidia-smi 可用）
        if command -v nvidia-smi &> /dev/null; then
            # 可选：检查 GPU 内存使用
            nvidia-smi --query-gpu=memory.used --format=csv,noheader
        fi
    done
done

echo ""
echo "========================================="
echo "所有测试完成"
echo "结束时间: $(date)"
echo "========================================="

# 生成汇总报告
echo ""
echo "生成测试汇总报告..."
echo "========================================="

REPORT_FILE="test_summary_${TIMESTAMP}.txt"
echo "测试汇总报告" > $REPORT_FILE
echo "生成时间: $(date)" >> $REPORT_FILE
echo "模型: $MODEL_PATH" >> $REPORT_FILE
echo "=========================================" >> $REPORT_FILE
echo "" >> $REPORT_FILE

for DATASET in "${DATASETS[@]}"; do
    echo "数据集: $DATASET" >> $REPORT_FILE
    echo "-----------------------------------------" >> $REPORT_FILE
    
    for CONFIG in "${CONFIGS[@]}"; do
        CONFIG_NAME=$(echo $CONFIG | cut -d'|' -f1)
        LOG_FILE="${LOG_DIR}/${DATASET}_${CONFIG_NAME}_${TIMESTAMP}.log"
        
        # 从日志中提取 pass@8 结果
        if [ -f "$LOG_FILE" ]; then
            # 提取 pass@8 值
            PASS_RESULT=$(grep -E '"pass@8": [0-9.]+' $LOG_FILE | tail -1 | sed 's/.*"pass@8": \([0-9.]*\).*/\1/')
            MEAN_RESULT=$(grep -E '"mean@8": [0-9.]+' $LOG_FILE | tail -1 | sed 's/.*"mean@8": \([0-9.]*\).*/\1/')
            
            if [ -n "$PASS_RESULT" ]; then
                echo "  $CONFIG_NAME: pass@8 = $PASS_RESULT, mean@8 = $MEAN_RESULT" >> $REPORT_FILE
            else
                echo "  $CONFIG_NAME: 未找到结果" >> $REPORT_FILE
            fi
        else
            echo "  $CONFIG_NAME: 日志文件不存在" >> $REPORT_FILE
        fi
    done
    echo "" >> $REPORT_FILE
done

echo "汇总报告已保存到: $REPORT_FILE"
cat $REPORT_FILE

# 可选：生成 CSV 格式报告
CSV_FILE="test_results_${TIMESTAMP}.csv"
echo "数据集,配置,pass@8,mean@8" > $CSV_FILE

for DATASET in "${DATASETS[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do
        CONFIG_NAME=$(echo $CONFIG | cut -d'|' -f1)
        LOG_FILE="${LOG_DIR}/${DATASET}_${CONFIG_NAME}_${TIMESTAMP}.log"
        
        if [ -f "$LOG_FILE" ]; then
            PASS_RESULT=$(grep -E '"pass@8": [0-9.]+' $LOG_FILE | tail -1 | sed 's/.*"pass@8": \([0-9.]*\).*/\1/')
            MEAN_RESULT=$(grep -E '"mean@8": [0-9.]+' $LOG_FILE | tail -1 | sed 's/.*"mean@8": \([0-9.]*\).*/\1/')
            
            if [ -z "$PASS_RESULT" ]; then
                PASS_RESULT="N/A"
                MEAN_RESULT="N/A"
            fi
        else
            PASS_RESULT="N/A"
            MEAN_RESULT="N/A"
        fi
        
        echo "$DATASET,$CONFIG_NAME,$PASS_RESULT,$MEAN_RESULT" >> $CSV_FILE
    done
done

echo "CSV 结果已保存到: $CSV_FILE"