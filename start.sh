#!/bin/bash

# --- 配置 ---
# 模型在您服务器上的本地路径
MODEL_PATH="/home/u2021110842/Qwen2.5-7B-Instruct"
# 您的主业务逻辑脚本路径
SCRIPT_PATH="/home/u2021110842/autodl-tmp/edge-star-test/infer-new-dataset.py"
# 日志和进程ID文件的存放目录
LOG_DIR="vllm_logs"
PID_FILE="pids.txt"
# vLLM服务暴露的模型名称
SERVED_NAME="qwen2.5-7b"
# 要启动的服务实例数量 (GPU数量)
NUM_GPUS=4

# --- 脚本正文 ---

# 创建日志目录
mkdir -p $LOG_DIR
echo "Starting VLLM services..."

# 清空旧的PID文件
> $PID_FILE

# 循环启动多个vLLM实例
for i in $(seq 0 $((3)))
do
    PORT=$((8000 + i))
    echo "Launching vLLM on GPU $i, Port $PORT..."
    
    # 使用nohup在后台启动vLLM服务，并将日志重定向
    CUDA_VISIBLE_DEVICES=$i nohup python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$SERVED_NAME" \
        --port "$PORT" \
        --trust-remote-code \
        --dtype auto \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.80 \
        --max-model-len 32768 > "$LOG_DIR/vllm_gpu${i}.log" 2>&1 &
    
    # 将新启动的进程ID记录到文件
    echo $! >> $PID_FILE
done

echo "-----------------------------------------------------"
echo "Waiting for all $NUM_GPUS vLLM services to become ready..."
ALL_READY=false
TOTAL_WAIT_SECONDS=0
MAX_WAIT_SECONDS=300

# 循环检查所有vLLM服务的健康状态
while [ "$ALL_READY" = false ]; do
    READY_COUNT=0
    # 精确地检查已启动的服务
    for i in $(seq 0 $((3))); do
        PORT=$((8000 + i))
        # 使用curl检查/health端点，--fail使其在失败时返回非零退出码
        if curl --silent --fail http://localhost:$PORT/health > /dev/null; then
            ((READY_COUNT++))
        fi
    done

    # 当就绪的服务数量等于我们启动的总数时，认为全部准备就绪
    if [ "$READY_COUNT" -eq "$NUM_GPUS" ]; then
        ALL_READY=true
        echo "All $NUM_GPUS vLLM services are up and running!"
    else
        echo "$READY_COUNT/$NUM_GPUS services ready. Waiting 5 more seconds..."
        sleep 5
        TOTAL_WAIT_SECONDS=$((TOTAL_WAIT_SECONDS + 5))
        # 超时检查
        if [ "$TOTAL_WAIT_SECONDS" -gt "$MAX_WAIT_SECONDS" ]; then
            echo "Error: Timed out waiting for vLLM services to start. Please check the logs in '$LOG_DIR'."
            exit 1
        fi
    fi
done
echo "-----------------------------------------------------"

# --- 启动主脚本 ---
echo "Starting main Python script..."
nohup python "$SCRIPT_PATH" > "$LOG_DIR/main_script.log" 2>&1 &
echo $! >> $PID_FILE

echo "All processes started successfully."
echo "You can now safely close this terminal."
echo "To stop all processes, run a command like: kill \$(cat pids.txt)"