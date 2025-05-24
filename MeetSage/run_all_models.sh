#!/bin/zsh

set -e

# 設置 Ollama 上下文長度
export OLLAMA_CONTEXT_LENGTH=8192

# 預設模型列表
MODELS=(
    "Phi4:latest"
    "cwchang/llama3-taide-lx-8b-chat-alpha1:latest"
    "gemma2:9b"
    "gemma3:12b"
    "llama3.1:8b"
    "mistral:7b"
    "phi3:latest"
)

ITERATIONS=${1:-3}

# 關閉舊的 Ollama 服務
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 停止任何運行的 Ollama 服務..."
pkill -f ollama || true
sleep 2

# 啟動 Ollama 服務
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 啟動 Ollama 服務..."
nohup ollama serve > ollama.log 2>&1 &
OLLAMA_PID=$!
sleep 10

# 檢查服務是否正常
if ! ollama list >/dev/null 2>&1; then
    echo "錯誤: Ollama 服務啟動失敗"
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
fi

# 啟動 Python 虛擬環境（如有）
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "已啟動虛擬環境"
fi

# 依序測試每個模型
for model in "${MODELS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 測試模型: $model"
    if ! ollama list | awk '{print $1}' | grep -Fxq "$model"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 錯誤: 模型 $model 未找到，跳過..."
        continue
    fi
    python test_prompt_optimization.py --model "$model" --iterations "$ITERATIONS"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 模型 $model 測試完成"
done

# 關閉 Ollama 服務
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 停止 Ollama 服務..."
kill $OLLAMA_PID 2>/dev/null || true
sleep 2

if [ -d ".venv" ]; then
    deactivate
    echo "已退出虛擬環境"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有測試完成！"
