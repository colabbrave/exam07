#!/bin/zsh

# 設置錯誤處理
set -e

# 顯示使用說明
show_help() {
    echo "使用方法: $0 [選項]"
    echo "選項:"
    echo "  --model MODEL_NAME     指定要使用的模型名稱"
    echo "  --iterations NUM      指定優化迭代次數 (預設: 10)"
    echo "  --help                顯示此幫助信息"
    exit 0
}

# 解析命令行參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "未知參數: $1"
            show_help
            exit 1
            ;;
    esac
done

# 1. 關閉所有舊的 Ollama 進程
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 停止任何運行的 Ollama 服務..."
pkill -f ollama || true
sleep 2

# 2. 啟動虛擬環境
cd "$(dirname "$0")"

if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "已啟動虛擬環境"
else
    echo "錯誤: 找不到 .venv 目錄，請先建立虛擬環境"
    exit 1
fi

# 3. 背景啟動 ollama 伺服器
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 啟動 Ollama 服務..."
nohup ollama serve > ollama.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama 伺服器啟動，PID: $OLLAMA_PID"

# 等待 Ollama 服務啟動
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 等待 Ollama 服務啟動..."
sleep 10

# 檢查 Ollama 是否正常運行
if ! ollama list >/dev/null 2>&1; then
    echo "錯誤: Ollama 服務啟動失敗"
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
fi

# 4. 執行 Python 腳本
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 開始執行測試..."

# 構建命令
PYTHON_CMD="python test_prompt_optimization.py"

# 添加可選參數
if [ -n "$MODEL" ]; then
    PYTHON_CMD="$PYTHON_CMD --model \"$MODEL\""
fi

if [ -n "$ITERATIONS" ]; then
    PYTHON_CMD="$PYTHON_CMD --iterations $ITERATIONS"
fi

# 執行命令
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 執行命令: $PYTHON_CMD"
eval $PYTHON_CMD

# 5. 關閉 ollama 服務
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 停止 Ollama 服務..."
kill $OLLAMA_PID 2>/dev/null || true
sleep 2

# 6. 離開虛擬環境
deactivate
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 已退出虛擬環境"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部流程完成"
