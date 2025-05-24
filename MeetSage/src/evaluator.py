import json
import logging
import jieba
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import numpy as np
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from bert_score import BERTScorer
from transformers import logging as transformers_logging

# 禁用 transformers 的警告
transformers_logging.set_verbosity_error()

# 確保下載必要的 NLTK 資料
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 jieba 分詞器
jieba.initialize()

@dataclass
class EvaluationResult:
    """評估結果數據類"""
    rouge1: Dict[str, float] = field(default_factory=dict)  # ROUGE-1 分數
    rouge2: Dict[str, float] = field(default_factory=dict)  # ROUGE-2 分數
    rougeL: Dict[str, float] = field(default_factory=dict)  # ROUGE-L 分數
    bleu: float = 0.0               # BLEU 分數
    bert_score: Dict[str, float] = field(default_factory=dict)  # BERTScore 分數
    avg_score: float = 0.0          # 平均分數 (0-1)

    
    def to_dict(self) -> Dict:
        """轉換為字典格式"""
        return {
            'rouge1': self.rouge1,
            'rouge2': self.rouge2,
            'rougeL': self.rougeL,
            'bleu': self.bleu,
            'bert_score': self.bert_score,
            'avg_score': self.avg_score
        }

class MeetingEvaluator:
    """會議紀錄評估器"""
    
    def __init__(self, device: str = None):
        """
        初始化評估器
        
        Args:
            device: 計算設備 ('cuda' 或 'cpu')，如果為 None 則自動選擇
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用設備: {self.device}")
        
        # 初始化 ROUGE
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        
        # 延遲加載 BERTScorer
        self._bertscorer = None
    
    @property
    def bertscorer(self):
        """延遲加載 BERTScorer"""
        if self._bertscorer is None:
            logger.info("正在加載 BERTScorer...")
            try:
                # 使用中文 BERT 模型
                self._bertscorer = BERTScorer(
                    model_type='bert-base-chinese',
                    lang='zh',
                    rescale_with_baseline=True,
                    device=self.device,
                    use_fast_tokenizer=True
                )
                logger.info("BERTScorer 加載完成")
            except Exception as e:
                logger.error(f"加載 BERTScorer 時出錯: {str(e)}")
                raise
        return self._bertscorer
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, Dict[str, float]]:
        """
        計算 ROUGE 分數 (支援中文)
        
        Args:
            reference: 參考文本
            candidate: 生成文本
            
        Returns:
            Dict: 包含 ROUGE-1, ROUGE-2, ROUGE-L 分數
        """
        try:
            # 確保輸入不為空
            if not reference or not candidate:
                raise ValueError("參考文本和生成文本都不能為空")
                
            # 使用 jieba 分詞後計算 ROUGE
            ref_tokens = ' '.join(jieba.cut(reference))
            can_tokens = ' '.join(jieba.cut(candidate))
            
            scores = self.rouge.get_scores(can_tokens, ref_tokens)
            return scores[0]  # 因為只比較一對
        except Exception as e:
            logger.error(f"計算 ROUGE 分數時出錯: {str(e)}")
            return {
                'rouge-1': {'f': 0, 'p': 0, 'r': 0},
                'rouge-2': {'f': 0, 'p': 0, 'r': 0},
                'rouge-l': {'f': 0, 'p': 0, 'r': 0}
            }
    
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """
        計算 BLEU 分數 (支援中文)
        
        Args:
            reference: 參考文本
            candidate: 生成文本
            
        Returns:
            float: BLEU 分數 (0-1)
        """
        try:
            # 使用 jieba 進行中文分詞
            ref_tokens = [list(jieba.cut(reference))]
            can_tokens = list(jieba.cut(candidate))
            
            # 計算 BLEU 分數 (使用 4-gram)
            return sentence_bleu(
                ref_tokens,
                can_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=self.smooth
            )
        except Exception as e:
            logger.error(f"計算 BLEU 分數時出錯: {str(e)}")
            return 0.0
    
    def calculate_bert_score(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        計算 BERTScore 分數
        
        Args:
            reference: 參考文本
            candidate: 生成文本
            
        Returns:
            Dict: 包含 precision, recall, f1 分數
        """
        try:
            # 確保輸入不為空
            if not reference or not candidate:
                raise ValueError("參考文本和生成文本都不能為空")
                
            # 計算 BERTScore
            with torch.no_grad():
                P, R, F1 = self.bertscorer.score([candidate], [reference])
            
            return {
                'precision': P.item(),
                'recall': R.item(),
                'f1': F1.item()
            }
        except Exception as e:
            logger.error(f"計算 BERTScore 時出錯: {str(e)}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def evaluate(self, reference: str, candidate: str) -> EvaluationResult:
        """
        評估生成的會議紀錄
        
        Args:
            reference: 參考答案 (人工撰寫的會議紀錄)
            candidate: 模型生成的會議紀錄
            
        Returns:
            EvaluationResult: 評估結果
        """
        # 計算 ROUGE 分數
        rouge_scores = self.calculate_rouge(reference, candidate)
        
        # 計算 BLEU 分數
        bleu_score = self.calculate_bleu(reference, candidate)
        
        # 計算 BERTScore
        bert_score = self.calculate_bert_score(reference, candidate)
        
        # 計算平均分數 (ROUGE-L F1, BLEU 和 BERTScore F1 的平均)
        avg_score = (rouge_scores['rouge-l']['f'] + bleu_score + bert_score['f1']) / 3
        
        return EvaluationResult(
            rouge1=rouge_scores['rouge-1'],
            rouge2=rouge_scores['rouge-2'],
            rougeL=rouge_scores['rouge-l'],
            bleu=bleu_score,
            bert_score=bert_score,
            avg_score=avg_score
        )

def save_evaluation_results(results: Dict, output_path: str):
    """儲存評估結果"""
    try:
        # 確保輸出目錄存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 將 dataclass 轉換為可序列化的字典
        serializable_results = {
            key: value.to_dict() if hasattr(value, 'to_dict') else value
            for key, value in results.items()
        }
        
        # 儲存為 JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"評估結果已儲存至 {output_path}")
    except Exception as e:
        logger.error(f"儲存評估結果時出錯: {str(e)}")
        raise

def load_evaluation_results(file_path: str) -> Dict:
    """載入評估結果"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"載入評估結果時出錯: {str(e)}")
        raise

def print_evaluation_summary(results: Dict):
    """打印評估結果摘要"""
    if not results:
        print("沒有可用的評估結果")
        return
    
    print("\n=== 評估結果摘要 ===")
    for prompt_name, metrics in results.items():
        if isinstance(metrics, dict) and 'avg_score' in metrics:
            print(f"\n{prompt_name}:")
            print(f"  平均分數: {metrics['avg_score']:.4f}")
            print(f"  ROUGE-1 F1: {metrics['rouge1']['f']:.4f}")
            print(f"  ROUGE-2 F1: {metrics['rouge2']['f']:.4f}")
            print(f"  ROUGE-L F1: {metrics['rougeL']['f']:.4f}")
            print(f"  BLEU: {metrics['bleu']:.4f}")

if __name__ == "__main__":
    # 使用範例
    evaluator = MeetingEvaluator()
    
    # 範例參考答案和生成結果
    reference = """
    會議紀錄
    日期：2025年5月20日
    與會者：張三、李四、王五
    
    討論事項：
    1. 專案進度落後一週
    2. 需要增加人手
    
    決議：
    1. 招募新成員加入團隊
    2. 下週一前完成招募
    """
    
    candidate = """
    會議紀錄
    日期：2025年5月20日
    與會者：張三、李四、王五
    
    討論事項：
    1. 專案進度落後一週
    2. 需要增加人手
    
    決議：
    1. 招募新成員
    2. 下週一前完成
    """
    
    # 評估
    result = evaluator.evaluate(reference, candidate)
    
    # 打印結果
    print("評估結果:")
    print(f"ROUGE-1 F1: {result.rouge1['f']:.4f}")
    print(f"ROUGE-2 F1: {result.rouge2['f']:.4f}")
    print(f"ROUGE-L F1: {result.rougeL['f']:.4f}")
    print(f"BLEU: {result.bleu:.4f}")
    print(f"平均分數: {result.avg_score:.4f}")
