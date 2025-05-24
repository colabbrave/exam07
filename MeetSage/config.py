import os
from pathlib import Path

# 基礎路徑設定
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
PROMPT_DIR = BASE_DIR / 'prompts'
EVAL_DIR = BASE_DIR / 'eval'

# 模型配置
DEFAULT_MODEL = 'phi3:latest'  # Ollama 模型名稱
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7

# 目錄創建
for directory in [DATA_DIR, MODEL_DIR, PROMPT_DIR, EVAL_DIR]:
    directory.mkdir(exist_ok=True)

# 日誌配置
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True
        },
    }
}
