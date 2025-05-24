import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import json

# 添加專案根目錄到 Python 路徑
sys.path.append(str(Path(__file__).parent))

from src.evaluator import MeetingEvaluator, save_evaluation_results, print_evaluation_summary

def load_test_case(test_case_dir: Path) -> Dict[str, str]:
    """載入測試案例"""
    if not test_case_dir.exists():
        raise FileNotFoundError(f"測試案例目錄不存在: {test_case_dir}")
    
    test_cases = {}
    for file in test_case_dir.glob("*.txt"):
        with open(file, 'r', encoding='utf-8') as f:
            test_cases[file.stem] = f.read()
    
    return test_cases

def evaluate_prompts(test_cases: Dict[str, Dict[str, str]], output_dir: Path):
    """評估多個提示詞"""
    evaluator = MeetingEvaluator()
    all_results = {}
    
    # 建立輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for case_name, case_data in test_cases.items():
        if 'reference' not in case_data or 'candidate' not in case_data:
            print(f"警告: 測試案例 {case_name} 缺少參考答案或生成結果，跳過")
            continue
        
        print(f"\n評估測試案例: {case_name}")
        print("-" * 50)
        
        # 評估
        result = evaluator.evaluate(case_data['reference'], case_data['candidate'])
        
        # 儲存結果
        all_results[case_name] = result.to_dict()
        
        # 打印摘要
        print(f"平均分數: {result.avg_score:.4f}")
        print(f"ROUGE-1 F1: {result.rouge1['f']:.4f}")
        print(f"ROUGE-2 F1: {result.rouge2['f']:.4f}")
        print(f"ROUGE-L F1: {result.rougeL['f']:.4f}")
        print(f"BLEU: {result.bleu:.4f}")
    
    # 儲存所有結果
    results_file = output_dir / "evaluation_results.json"
    save_evaluation_results(all_results, str(results_file))

    # 額外存檔：將每次評估的詳細結果另存一份（加上日期時間）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_file = Path(output_dir) / f"evaluation_results_{timestamp}.json"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"詳細結果已存檔：{detailed_file}")

    # 打印摘要
    print("\n=== 所有測試案例評估完成 ===")
    print(f"結果已儲存至: {results_file}")
    print_evaluation_summary(all_results)

def main():
    # 範例測試案例
    test_cases = {
        "case1": {
            "reference": """
            會議紀錄
            日期：2025年5月20日
            與會者：張三、李四、王五
            
            討論事項：
            1. 專案進度落後一週
            2. 需要增加人手
            
            決議：
            1. 招募新成員加入團隊
            2. 下週一前完成招募
            """,
            "candidate": """
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
        }
    }
    
    # 評估提示詞
    evaluate_prompts(test_cases, Path("eval/results"))

if __name__ == "__main__":
    main()
