import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
import json
import ollama
import asyncio
from datetime import datetime

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeetingMiner:
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name

    def generate_minutes(self, transcript: str, prompt_template: str) -> dict:
        """生成會議紀錄"""
        try:
            # 確保提示詞中包含 {transcript} 佔位符
            if "{transcript}" not in prompt_template:
                prompt_template = "{transcript}\n\n" + prompt_template
                
            # 替換提示詞中的佔位符
            prompt = prompt_template.format(transcript=transcript)
            
            # 添加日誌，方便調試
            logger.debug(f"發送給模型的提示詞長度: {len(prompt)} 字元")
            if len(prompt) > 1000:  # 只記錄前1000個字元，避免日誌過大
                logger.debug(f"提示詞開頭: {prompt[:1000]}...")
            else:
                logger.debug(f"提示詞: {prompt}")
            
            # 發送請求到 Ollama
            response = ollama.chat(
                model=self.model_name, 
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.3,  # 降低隨機性，使輸出更穩定
                    'top_p': 0.9,
                    'max_tokens': 2000  # 限制生成的最大token數量
                }
            )
            
            # 記錄回應長度
            content = response['message']['content']
            logger.debug(f"收到回應，長度: {len(content)} 字元")
            
            return {"content": content, "status": "success"}
            
        except Exception as e:
            logger.error(f"生成會議紀錄時發生錯誤：{str(e)}")
            logger.exception("詳細錯誤信息:")  # 記錄完整的堆疊追蹤
            return {"content": "", "status": "error", "error": str(e)}

    def load_transcript(self, file_path: str) -> str:
        """載入會議逐字稿"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"載入逐字稿失敗: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """預處理會議逐字稿"""
        # 移除多餘的空白字元
        text = ' '.join(text.split())
        # 簡單的標點符號正規化
        text = re.sub(r'([，。；：、])', r'\1 ', text)
        return text.strip()
    
    def save_minutes(self, minutes: Dict, output_path: str):
        """儲存會議紀錄"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(minutes, f, ensure_ascii=False, indent=2)
            logger.info(f"會議紀錄已儲存至 {output_path}")
        except Exception as e:
            logger.error(f"儲存會議紀錄失敗: {str(e)}")
            raise

def main():
    # 使用範例
    miner = MeetingMiner()
    
    # 載入範例逐字稿
    transcript = """
    會議時間：2025年5月20日 14:00-15:30
    與會人員：張三、李四、王五
    
    張三：我們今天要討論專案進度。
    李四：目前開發進度落後一週。
    王五：我們需要調整時程或增加人手。
    """
    
    # 生成會議紀錄
    minutes = miner.generate_minutes(
        transcript=transcript,
        prompt_template="請幫我根據以下會議逐字稿生成一份會議紀錄：\n\n{transcript}"
    )
    
    # 顯示結果
    print("=== 生成的會議紀錄 ===")
    print(minutes['content'])
    
    # 儲存結果
    output_dir = Path('data/output')
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"meeting_minutes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    miner.save_minutes(minutes, str(output_path))

if __name__ == "__main__":
    main()
