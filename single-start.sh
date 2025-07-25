echo "--- 初始化配置 ---"

# --- 核心配置区 ---

# 1. 服务与模型配置
MODEL_PATH="/home/u2021110842/Qwen2.5-7B-Instruct" 
MODEL_NAME="qwen2.5-7b"
PORT=8000
GPU_ID=0 # <<< 新增: 指定使用的GPU ID

# 2. 脚本与执行配置
PIPELINE_SCRIPT_PATH="/home/u2021110842/autodl-tmp/edge-star-test/single-infer.py" 
MAX_WORKERS=12

# 3. 数据文件路径
BGP_PATHS_INPUT_FILE="/home/u2021110842/autodl-tmp/20250211data/new_datasettop250.txt"
AS_ORG_DB_FILE="/home/u2021110842/autodl-tmp/20250211data/asn_organization_mapping.json"
GROUND_TRUTH_FILE="/home/u2021110842/autodl-tmp/20250211data/as_relationships_cache—update.json"

# 4. 输出目录
OUTPUT_DIR="LLM_Final_Results_$(date +%Y%m%d)_${MODEL_NAME}"

# 5. [可选] 快速测试模式
# LIMIT_COUNT=100

# --- 执行区 ---

# 准备工作
LIMIT_ARG=""
if [ -n "$LIMIT_COUNT" ]; then
    echo "⚠️  快速测试模式已启用，将只处理 ${LIMIT_COUNT} 条路径。"
    LIMIT_ARG="--limit ${LIMIT_COUNT}"
fi

LOG_DIR="vllm_logs_$(date +%Y%m%d)"
PID_FILE="${LOG_DIR}/pids.txt"
mkdir -p $LOG_DIR
> $PID_FILE

# 1. 启动 VLLM 服务
echo "--- [步骤 1/4] 正在后台启动 VLLM 服务... ---"
CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -m vllm.entrypoints.openai.api_server \
   --model "$MODEL_PATH" \
   --served-model-name "$MODEL_NAME" \
   --port "$PORT" \
   --dtype auto \
   --tensor-parallel-size 1 > "${LOG_DIR}/vllm_service.log" 2>&1 &

VLLM_PID=$!
echo $VLLM_PID > $PID_FILE
echo "VLLM 服务已启动，进程ID: ${VLLM_PID}，日志文件: ${LOG_DIR}/vllm_service.log"

# <<< 新增：VLLM 健康检查与智能等待 >>>
echo "--- [步骤 2/4] 正在等待 VLLM 服务就绪... ---"
MAX_WAIT_SECONDS=300 # 设置最长等待时间为300秒 (5分钟)
SECONDS_WAITED=0

while true; do
    # 使用 curl 的 -s (静默) 和 -f (失败时返回错误码) 选项
    if curl -sf http://localhost:${PORT}/health > /dev/null; then
        echo "✅ VLLM 服务已在端口 ${PORT} 上准备就绪！"
        break
    else
        if [ $SECONDS_WAITED -ge $MAX_WAIT_SECONDS ]; then
            echo "❌ 错误：等待 VLLM 服务超时（超过 ${MAX_WAIT_SECONDS} 秒）。"
            echo "请检查日志文件 '${LOG_DIR}/vllm_service.log' 以确定问题。"
            kill $VLLM_PID # 尝试杀死启动失败的进程
            exit 1
        fi
        echo "服务尚未就绪，5秒后重试... (已等待 ${SECONDS_WAITED}s)"
        sleep 5
        SECONDS_WAITED=$((SECONDS_WAITED + 5))
    fi
done
# <<< 智能等待结束 >>>

# 3. 调用主流程 Python 脚本
echo "--- [步骤 3/4] VLLM 已就绪，开始执行主流程脚本... ---"
python "$PIPELINE_SCRIPT_PATH" \
    --port "$PORT" \
    --model-name "$MODEL_NAME" \
    --max-workers "$MAX_WORKERS" \
    --bgp-paths-input "$BGP_PATHS_INPUT_FILE" \
    --as-org-db "$AS_ORG_DB_FILE" \
    --ground-truth "$GROUND_TRUTH_FILE" \
    --output-dir "$OUTPUT_DIR" \
    $LIMIT_ARG

# 4. 结束
if [ $? -eq 0 ]; then
    echo "--- [步骤 4/4] 流程执行完毕 ---"
    echo "🎉 Python 脚本已成功执行完毕！"
    echo "所有结果和报告请查看目录: ${OUTPUT_DIR}"
    echo "如需停止VLLM服务，请运行: kill ${VLLM_PID}"
else
    echo "--- [步骤 4/4] 流程执行失败 ---"
    echo "❌ Python 脚本执行失败，请检查上面的错误信息。"
    echo "VLLM服务仍在后台运行，进程ID: ${VLLM_PID}。如果需要，请手动停止。"
fi