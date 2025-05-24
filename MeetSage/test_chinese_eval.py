import sys
from pathlib import Path
from src.evaluator import MeetingEvaluator

def test_chinese_evaluation():
    """測試中文評估系統"""
    evaluator = MeetingEvaluator()
    
    # 測試案例 1: 完全相同的文本
    ref1 = "這是一個測試句子。"
    cand1 = "這是一個測試句子。"
    
    # 測試案例 2: 相似但不完全相同的文本
    ref2 = "今天天氣真好，我們去公園散步吧！"
    cand2 = "今天天氣不錯，我們去公園走走吧！"
    
    # 測試案例 3: 差異較大的文本
    ref3 = "人工智慧是當今最熱門的技術之一。"
    cand3 = "機器學習正在改變我們的生活。"
    
    # 評估測試案例
    def evaluate_case(reference, candidate, case_name):
        print(f"\n{case_name}")
        print("-" * 50)
        print(f"參考文本: {reference}")
        print(f"生成文本: {candidate}")
        
        result = evaluator.evaluate(reference, candidate)
        
        print("\n評估結果:")
        print("--- 傳統指標 ---")
        print(f"ROUGE-1 F1: {result.rouge1['f']:.4f}")
        print(f"ROUGE-2 F1: {result.rouge2['f']:.4f}")
        print(f"ROUGE-L F1: {result.rougeL['f']:.4f}")
        print(f"BLEU: {result.bleu:.4f}")
        print("\n--- BERTScore ---")
        print(f"Precision: {result.bert_score['precision']:.4f}")
        print(f"Recall:    {result.bert_score['recall']:.4f}")
        print(f"F1:        {result.bert_score['f1']:.4f}")
        print("\n--- 綜合分數 ---")
        print(f"平均分數: {result.avg_score:.4f}")
    
    # 執行測試案例
    evaluate_case(ref1, cand1, "測試案例 1: 完全相同的文本")
    evaluate_case(ref2, cand2, "測試案例 2: 相似的文本")
    evaluate_case(ref3, cand3, "測試案例 3: 差異較大的文本")

if __name__ == "__main__":
    # 添加專案根目錄到 Python 路徑
    sys.path.append(str(Path(__file__).parent))
    
    # 執行測試
    test_chinese_evaluation()
