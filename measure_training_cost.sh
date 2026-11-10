#!/bin/bash
# Measure training cost for LEPO and DLaR (GRPO already run separately)
# Usage: bash scripts/measure_training_cost.sh

cd /public/home/xiangyuduan/xcluo/lepo-main

# ===== Configurations =====
ACCELERATE_CONFIG="configs/accelerate/ds_zero2.yaml"
NUM_PROCESSES=2
OUTPUT_DIR="./cost_measurements"
mkdir -p "${OUTPUT_DIR}"

GRPO_CONFIG="configs/grpo/llama3_1b_grpo.yaml"
LEPO_CONFIG="configs/lepo/llama3_1b_latent32_gumbel050.yaml"
DLAR_CONFIG="configs/dlar/llama3_1b_latent32_gumbel050.yaml"

MIN_VALID_SECONDS=600   # 低于此值视为训练失败

# ===== Helper: run one experiment =====
run_experiment() {
    local NAME=$1
    local SCRIPT=$2
    local CONFIG=$3
    local REWARD_TYPE=$4

    echo ""
    echo "============================================================"
    echo ">>> Running ${NAME}"
    echo "    Script: ${SCRIPT}"
    echo "    Config: ${CONFIG}"
    echo "============================================================"

    # 检查文件是否存在
    if [ ! -f "${SCRIPT}" ]; then
        echo ">>> ERROR: script not found: ${SCRIPT}"
        return 1
    fi
    if [ ! -f "${CONFIG}" ]; then
        echo ">>> ERROR: config not found: ${CONFIG}"
        return 1
    fi

    local MEM_LOG="${OUTPUT_DIR}/${NAME}_memory.csv"
    local TIME_LOG="${OUTPUT_DIR}/${NAME}_time.txt"
    local PEAK_LOG="${OUTPUT_DIR}/${NAME}_peak_mem.txt"
    local TRAIN_LOG="${OUTPUT_DIR}/${NAME}_stdout.log"

    # 后台用 nvidia-smi 采样（每 1 秒）
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total \
        --format=csv -l 1 > "${MEM_LOG}" &
    local MONITOR_PID=$!

    local START_TIME=$(date +%s)

    # 运行训练，同时保存 stdout 以便解析 Trainer 日志
    accelerate launch --config_file "${ACCELERATE_CONFIG}" \
        --num_processes "${NUM_PROCESSES}" \
        "${SCRIPT}" \
        --config "${CONFIG}" \
        --reward_type "${REWARD_TYPE}" 2>&1 | tee "${TRAIN_LOG}"
    local EXIT_CODE=${PIPESTATUS[0]}

    local END_TIME=$(date +%s)
    local ELAPSED=$((END_TIME - START_TIME))

    kill ${MONITOR_PID} 2>/dev/null || true
    sleep 1

    # 检查退出码
    if [ "${EXIT_CODE}" -ne 0 ]; then
        echo ">>> ERROR: ${NAME} exited with code ${EXIT_CODE}"
    fi

    # 保存时间
    echo "Training time (seconds): ${ELAPSED}" > "${TIME_LOG}"
    echo "Training time (hours): $(echo "scale=3; ${ELAPSED}/3600" | bc)" >> "${TIME_LOG}"

    # 从 CSV 提取显存峰值
    python3 - "${MEM_LOG}" "${PEAK_LOG}" <<'PYEOF'
import csv, sys
from collections import defaultdict

mem_log, peak_log = sys.argv[1], sys.argv[2]
peaks = defaultdict(float)
with open(mem_log) as f:
    for row in csv.DictReader(f):
        try:
            idx = int(row[" index"].strip())
            used = float(row[" memory.used [MiB]"].strip().replace(" MiB", ""))
            peaks[idx] = max(peaks[idx], used)
        except (ValueError, KeyError):
            continue

if peaks:
    with open(peak_log, "w") as f:
        for idx in sorted(peaks):
            f.write(f"GPU {idx}: {peaks[idx]:.0f} MB ({peaks[idx]/1024:.2f} GB)\n")
        f.write(f"Max peak: {max(peaks.values()):.0f} MB ({max(peaks.values())/1024:.2f} GB)\n")
    print(f"  nvidia-smi peak (approx): {max(peaks.values())/1024:.2f} GB")
else:
    print("  WARNING: no memory samples collected")
PYEOF

    # 从 Trainer 日志解析 train_runtime
    if grep -q "train_runtime" "${TRAIN_LOG}"; then
        echo "" >> "${TIME_LOG}"
        echo "From Trainer log:" >> "${TIME_LOG}"
        grep -o "'train_runtime': [0-9.]*" "${TRAIN_LOG}" | tail -1 >> "${TIME_LOG}" || true
    fi

    # 合理性检查
    if [ "${ELAPSED}" -lt "${MIN_VALID_SECONDS}" ]; then
        echo ">>> WARNING: ${NAME} finished in ${ELAPSED}s (< ${MIN_VALID_SECONDS}s), likely FAILED"
    fi

    echo ">>> ${NAME} completed in ${ELAPSED} seconds (exit=${EXIT_CODE})"
    echo "    Time log:   ${TIME_LOG}"
    echo "    Memory log: ${MEM_LOG}"
    echo "    Peak mem:   ${PEAK_LOG}"
    echo "    Stdout log: ${TRAIN_LOG}"
}

# ===== Run experiments =====
# GRPO 已在单独运行，这里跳过，避免 GPU 冲突
run_experiment "GRPO" "scripts/train_grpo.py" "${GRPO_CONFIG}" "standard"
run_experiment "LEPO" "scripts/train_lepo.py" "${LEPO_CONFIG}" "standard"
#run_experiment "DLaR" "scripts/train_dlar.py" "${DLAR_CONFIG}" "standard"

# ===== 汇总 =====
echo ""
echo "============================================================"
echo "All experiments completed!"
echo "Results saved in: ${OUTPUT_DIR}"
echo "============================================================"

python3 - "${OUTPUT_DIR}" <<'PYEOF'
import os, re, sys

out_dir = sys.argv[1]
rows = []
for name in ["GRPO", "LEPO", "DLaR"]:
    time_file = os.path.join(out_dir, f"{name}_time.txt")
    peak_file = os.path.join(out_dir, f"{name}_peak_mem.txt")
    if not os.path.exists(time_file):
        continue
    hours = None
    with open(time_file) as f:
        for line in f:
            m = re.search(r"hours\):\s*([0-9.]+)", line)
            if m:
                hours = float(m.group(1))
                break
    mem_gb = None
    if os.path.exists(peak_file):
        with open(peak_file) as f:
            for line in f:
                m = re.search(r"Max peak:.*\(([0-9.]+) GB\)", line)
                if m:
                    mem_gb = float(m.group(1))
                    break
    rows.append((name, hours, mem_gb))

print(f"{'Method':<8} {'Time (h)':>10} {'Mem (GB)':>10}")
print("-" * 30)
for name, h, m in rows:
    hs = f"{h:.3f}" if h is not None else "N/A"
    ms = f"{m:.2f}" if m is not None else "N/A"
    print(f"{name:<8} {hs:>10} {ms:>10}")

# Overhead relative to GRPO
grpo = next((r for r in rows if r[0] == "GRPO"), None)
dlar = next((r for r in rows if r[0] == "DLaR"), None)
if grpo and dlar and grpo[1] and grpo[2] and dlar[1] and dlar[2]:
    t_oh = (dlar[1] - grpo[1]) / grpo[1] * 100
    m_oh = (dlar[2] - grpo[2]) / grpo[2] * 100
    print(f"\nRollout Overhead (DLaR vs GRPO): +{t_oh:.1f}% time / +{m_oh:.1f}% mem")
PYEOF