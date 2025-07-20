#!/bin/bash

# --- Configuration ---
# 模型在您服务器上的本地路径
MODEL_PATH="/home/u2021110842/Qwen2.5-7B-Instruct"

# 您的主业务逻辑脚本路径
SCRIPT_PATH="/home/u2021110842/autodl-tmp/edge-star-test/single-infer.py"

# <<< 修改: 在此处定义数据文件路径
INPUT_FILE="/home/u2021110842/autodl-tmp/BGPAgent/20250211data/new_dataset.txt"
AS_DB_FILE="/home/u2021110842/autodl-tmp/BGPAgent/20250211data/asn_organization_mapping.json"

# 日志和进程ID文件的存放目录
LOG_DIR="vllm_logs"
PID_FILE="pids.txt"

# vLLM服务暴露的模型名称
SERVED_NAME="qwen2.5-7b"

# --- Single-GPU Setup ---
# 明确指定端口和使用的GPU ID
PORT=8000
GPU_ID=0


# --- 脚本正文 (VLLM启动部分无变化) ---

echo "--- Starting VLLM Service Setup ---"
mkdir -p $LOG_DIR
echo "Log directory is set to: $LOG_DIR"
> $PID_FILE
echo "Old PID file cleared."
echo "Launching vLLM on GPU $GPU_ID, Port $PORT..."

CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --trust-remote-code \
    --dtype auto \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 32768 > "$LOG_DIR/vllm_gpu_${GPU_ID}.log" 2>&1 &

VLLM_PID=$!
echo $VLLM_PID >> $PID_FILE
echo "VLLM service started with PID: $VLLM_PID. Log file: $LOG_DIR/vllm_gpu_${GPU_ID}.log"

echo "-----------------------------------------------------"
echo "Waiting for the VLLM service to become ready..."
TOTAL_WAIT_SECONDS=0
MAX_WAIT_SECONDS=300

# Health check loop for the single service instance
while true; do
    # Use curl to check the /health endpoint. --fail makes it return an error code on failure.
    if curl --silent --fail http://localhost:$PORT/health > /dev/null; then
        echo "VLLM service on port $PORT is up and running!"
        break
    else
        sleep 5
        TOTAL_WAIT_SECONDS=$((TOTAL_WAIT_SECONDS + 5))
        # Timeout check
        if [ "$TOTAL_WAIT_SECONDS" -gt "$MAX_WAIT_SECONDS" ]; then
            echo "Error: Timed out waiting for the VLLM service to start."
            echo "Please check the log: '$LOG_DIR/vllm_gpu_${GPU_ID}.log'"
            # Optional: kill the failed process before exiting
            kill $VLLM_PID
            exit 1
        fi
        echo "Service not ready yet. Waited $TOTAL_WAIT_SECONDS seconds..."
    fi
done
echo "-----------------------------------------------------"

# --- 启动主脚本 ---
echo "Starting main Python inference script..."

# <<< 核心修改: 调用Python脚本时，传入新增的文件路径参数
nohup python "$SCRIPT_PATH" \
    --model-name "$SERVED_NAME" \
    --port "$PORT" \
    --input-file "$INPUT_FILE" \
    --as-info-db "$AS_DB_FILE" > "$LOG_DIR/main_script.log" 2>&1 &

# 记录主脚本的PID
SCRIPT_PID=$!
echo $SCRIPT_PID >> $PID_FILE
echo "Main script started with PID: $SCRIPT_PID. Log file: $LOG_DIR/main_script.log"
echo "-----------------------------------------------------"

echo "All processes started successfully."
echo "You can now safely close this terminal."
echo "To monitor logs, use commands like: tail -f $LOG_DIR/main_script.log"
echo "To stop all processes, run the command: kill \$(cat $PID_FILE)"
