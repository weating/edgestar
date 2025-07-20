PID_FILE="pids.txt"

if [ -f "$PID_FILE" ]; then
    echo "Found PID file ($PID_FILE). Stopping processes by PID..."
    

    while read -r pid; do
        if ps -p "$pid" > /dev/null; then
            echo "Stopping process with PID: $pid"
            kill "$pid"
        else
            echo "Process with PID $pid not found. It might have already stopped."
        fi
    done < <(tac "$PID_FILE")

    sleep 2

    echo "Cleaning up PID file..."
    rm "$PID_FILE"
    
    echo "Stop script finished. All processes recorded in '$PID_FILE' have been sent a stop signal."


else
    echo "PID file '$PID_FILE' not found."
    echo "Attempting to stop processes by name as a fallback..."
    echo "This may stop other unrelated processes if their names match."
    

    pkill -f "vllm.entrypoints.openai.api_server"
    pkill -f "process_max_throughput.py" 

    echo
    echo "Sent kill signals to processes matching vLLM server and main script names."
    echo "Please use 'ps -ef | grep python' to verify all related processes have been stopped."
fi