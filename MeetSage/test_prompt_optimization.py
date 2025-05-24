"""
測試提示詞優化

這個腳本展示如何使用 PromptOptimizer 來優化會議記錄生成的提示詞。
"""
import os
import sys
import json
import time
import random
import jieba
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 導入 MeetingMiner 類
from src.meeting_miner import MeetingMiner

# 設置日誌
import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('meeting_miner.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_ollama_models():
    """取得本機 Ollama 所有模型名稱清單"""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m['name'] for m in data.get('models', [])]
    except Exception as e:
        print(f"取得 Ollama 模型失敗: {e}")
        return []

def load_test_cases():
    """
    載入所有逐字稿，並嘗試配對 reference 目錄中的會議紀錄。
    - 逐字稿文件名格式：XXX逐字稿.txt
    - 參考文件格式：XXX會議紀錄.txt
    """
    from pathlib import Path

    test_cases = []
    transcripts_dir = Path("data/transcripts")
    references_dir = Path("data/reference")

    if not transcripts_dir.exists():
        print(f"錯誤：找不到 transcripts 目錄")
        return []

    # 建立參考文件映射
    ref_files = {}
    if references_dir.exists():
        for ref_file in references_dir.glob("*會議紀錄*.txt"):
            base_name = ref_file.stem.replace("會議紀錄", "").strip()
            ref_files[base_name] = ref_file

    # 處理逐字稿文件
    for transcript_file in transcripts_dir.glob("*逐字稿*.txt"):
        base_name = transcript_file.stem.replace("逐字稿", "").strip()
        transcript = transcript_file.read_text(encoding="utf-8").strip()
        reference_file = ref_files.get(base_name)
        reference = None
        if reference_file:
            reference = reference_file.read_text(encoding="utf-8").strip()
            print(f"已配對: {transcript_file.name} <-> {reference_file.name}")
        else:
            print(f"未配對: {transcript_file.name} 找不到對應會議紀錄")

        test_cases.append({
            "name": transcript_file.stem,
            "transcript": transcript,
            "reference": reference,
            "transcript_file": transcript_file.name,
            "reference_file": reference_file.name if reference_file else None
        })

    if not test_cases:
        print("警告：沒有找到任何逐字稿")
    else:
        print(f"共找到 {len(test_cases)} 份逐字稿")

    return test_cases


def ask_user_choice(prompt, options, auto_mode=False, default=None):
    if auto_mode:
        # 自動模式下直接選第一個或指定預設
        return default if default is not None else options[0]
    print(prompt)
    for idx, opt in enumerate(options, 1):
        print(f"{idx}. {opt}")
    while True:
        try:
            choice = int(input("請輸入選項編號: "))
            if 1 <= choice <= len(options):
                return options[choice-1]
        except Exception:
            pass
        print("請輸入有效的選項編號。")

def ask_user_int(prompt, default, auto_mode=False):
    if auto_mode:
        return default
    val = input(f"{prompt} (預設: {default}): ")
    if val.strip() == '':
        return default
    try:
        return int(val)
    except Exception:
        print("輸入無效，使用預設值。")
        return default

def ask_user_float(prompt, default, auto_mode=False):
    if auto_mode:
        return default
    val = input(f"{prompt} (預設: {default}): ")
    if val.strip() == '':
        return default
    try:
        return float(val)
    except Exception:
        print("輸入無效，使用預設值。")
        return default

def evaluate(generated: str, reference: str) -> dict:
    """
    評估生成的會議紀錄與參考會議紀錄的相似度
    返回包含多個評估指標的字典
    """
    # 分詞
    gen_words = list(jieba.lcut(generated))
    ref_words = list(jieba.lcut(reference))
    
    # 轉換為集合用於計算重疊
    gen_set = set(gen_words)
    ref_set = set(ref_words)
    
    # 計算基本統計
    gen_len = len(gen_words)
    ref_len = len(ref_words)
    
    # 計算重疊
    intersection = len(gen_set.intersection(ref_set))
    union = len(gen_set.union(ref_set))
    
    # 計算各種相似度指標
    jaccard = intersection / union if union > 0 else 0.0
    
    # 計算重疊詞比例
    overlap_ratio = (2 * intersection) / (len(gen_set) + len(ref_set)) if (len(gen_set) + len(ref_set)) > 0 else 0.0
    
    # 計算長度比例
    length_ratio = min(gen_len, ref_len) / max(gen_len, ref_len) if max(gen_len, ref_len) > 0 else 0.0
    
    # 計算最終分數（加權平均）
    final_score = 0.5 * jaccard + 0.3 * overlap_ratio + 0.2 * length_ratio
    
    return {
        'score': final_score,  # 主要分數（0-1）
        'jaccard': jaccard,    # Jaccard 相似度
        'overlap_ratio': overlap_ratio,  # 重疊詞比例
        'length_ratio': length_ratio,    # 長度比例
        'gen_length': gen_len,  # 生成文本長度
        'ref_length': ref_len   # 參考文本長度
    }

def save_scores_csv(results: dict, output_path: str):
    """將所有分數記錄成 csv 檔"""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 自動建立目錄
    # 收集所有指標欄位
    all_metrics = set()
    for v in results.values():
        if isinstance(v, dict) and 'metrics' in v:
            all_metrics.update(v['metrics'].keys())
    fieldnames = ['variant', 'description'] + sorted(all_metrics)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, v in results.items():
            row = {'variant': name, 'description': v.get('description', '')}
            if 'metrics' in v:
                row.update(v['metrics'])
            writer.writerow(row)

def main():
    # 取得所有本機 Ollama 模型
    model_list = get_ollama_models()
    if not model_list:
        print("找不到任何 Ollama 模型，請先用 ollama pull 下載模型。"); return
    model = ask_user_choice("請選擇 Ollama 模型：", model_list)
    mode = ask_user_choice("選擇優化方式：", ["迭代次數", "限時(分鐘)"])

    # 初始化會議記錄生成器、評估器、測試案例、優化器
    logger.info(f"初始化會議記錄生成器... (模型: {model})")
    meeting_miner = MeetingMiner(model_name=model)
    logger.info("初始化評估器...")
    evaluator = MeetingEvaluator()
    test_cases = load_test_cases()
    logger.info(f"已載入 {len(test_cases)} 個測試案例")
    
    logger.info("初始化提示詞優化器...")
    optimizer = PromptOptimizer(
        evaluator=evaluator,
        meeting_miner=meeting_miner,
        test_cases=test_cases,
        model_name=model
    )

    logger.info("開始優化提示詞...")

    # 完全自動化：有 iterations 參數就自動執行
    auto_mode = False
    if args and hasattr(args, 'iterations') and args.iterations:
        auto_mode = True
        mode = "迭代次數"
        max_iterations = args.iterations
        logger.info(f"自動批次模式，設置最大迭代次數: {max_iterations}")
        min_improvement = 0.01
    else:
        mode = ask_user_choice("選擇優化方式：", ["迭代次數", "限時(分鐘)"], auto_mode=auto_mode, default="迭代次數")
        if mode == "迭代次數":
            max_iterations = ask_user_int("請輸入最大迭代次數 (預設: 10)", 10, auto_mode=auto_mode)
            min_improvement = ask_user_float("請輸入最小提升幅度 (如 0.01=1%) (預設: 0.01)", 0.01, auto_mode=auto_mode)
        else:
            max_iterations = 1
            min_improvement = ask_user_float("請輸入最小提升幅度 (如 0.01=1%) (預設: 0.01)", 0.01, auto_mode=auto_mode)

    result = optimizer.optimize(max_iterations=max_iterations, min_improvement=min_improvement)
    all_results = result['all_results']
    best_variant = result['best_variant']
    best_score = result['best_score']

    if mode == "限時(分鐘)":
        time_limit = input("請輸入限時(分鐘): ")
        time_limit = float(time_limit) if time_limit else 10
        min_improvement = input(f"請輸入最小提升幅度 (如 0.01=1%) (預設: 0.01): ")
        min_improvement = float(min_improvement) if min_improvement else 0.01
        
        start_time = time.time()
        iteration = 0
        best_score = 0
        best_variant = None
        all_results = {}
        
        while (time.time() - start_time) < time_limit * 60:
            result = optimizer.optimize(max_iterations=1, min_improvement=min_improvement)
            # 更新結果
            if result['all_results']:
                all_results.update(result['all_results'])
            # 評估結果
            eval_result = evaluate(result["content"], test_cases[0]["reference"])
            score = eval_result['score']
            results[test_cases[0]["name"]][iteration] = score
            
            # 輸出詳細評估結果
            print(f"\n=== 測試案例 '{test_cases[0]['name']}' ===")
            print(f"得分: {score:.4f} (Jaccard: {eval_result['jaccard']:.4f}, "
                  f"重疊率: {eval_result['overlap_ratio']:.4f}, "
                  f"長度比: {eval_result['length_ratio']:.4f})")
            print(f"生成長度: {eval_result['gen_length']}, 參考長度: {eval_result['ref_length']}")
            
            # 記錄詳細結果
            for key, value in eval_result.items():
                if key != 'score':
                    metric_key = f"{key}_{test_cases[0]['name']}"
                    if metric_key not in results:
                        results[metric_key] = [0.0] * len(prompt_templates)
                    results[metric_key][iteration] = value
            iteration += 1
        
        print(f"限時 {time_limit} 分鐘內共執行 {iteration} 次優化。")

    # 僅輸出 dict 結果，避免型別錯誤
    if not isinstance(all_results, dict):
        print("all_results 型別錯誤，無法輸出。")
        return
        
    print("\n最佳提示詞結果：")
    if best_variant:
        print(f"最佳分數：{best_score:.4f}")
        print("最佳提示詞：")
        print(best_variant)
    else:
        print("未找到有效的提示詞。")
    
    # 新增：存成 csv
    csv_path = f"prompt_optimization_results/{model}/scores.csv"
    save_scores_csv(all_results, csv_path)
    print(f"所有分數已存成 CSV：{csv_path}")
    
    # 輸出前三名
    print("\n前三名提示詞及分數：")
    sorted_items = sorted(
        [(k, v) for k, v in all_results.items() if isinstance(v, dict) and 'metrics' in v],
        key=lambda x: x[1]['metrics'].get('avg_score', 0),
        reverse=True
    )
    for rank, (name, info) in enumerate(sorted_items[:3], 1):
        print(f"第{rank}名: {name} 分數: {info['metrics'].get('avg_score', 0):.4f}")
        print(f"提示詞：\n{info['template']}\n")
        # 產生會議紀錄內容
        import time
        start_time = time.time()
        result_minutes = meeting_miner.generate_minutes(
            transcript=test_cases[0]['transcript'],
            prompt_template=info['template']
        )
        end_time = time.time()
        exec_time = end_time - start_time
        logger.info(f"模型: {model}, Prompt: {name}, 生成會議紀錄耗時: {exec_time:.2f} 秒")
        print(f"生成的會議紀錄：\n{result_minutes['content']}\n{'='*40}")
        # 將執行時間寫入 all_results
        if 'metrics' in info:
            info['metrics']['exec_time_sec'] = exec_time
        else:
            info['metrics'] = {'exec_time_sec': exec_time}
        all_results[name] = info

def stop_ollama():
    """停止 Ollama 服務"""
    import subprocess
    import time
    
    try:
        logger.info("正在停止 Ollama 服務...")
        subprocess.run(["pkill", "-f", "ollama"], check=False)
        time.sleep(1)  # 給進程一些時間來完全停止
        logger.info("Ollama 服務已停止")
        return True
    except Exception as e:
        logger.warning(f"停止 Ollama 服務時出錯: {e}")
        return False

def start_ollama():
    """在後台啟動 Ollama 服務並等待其準備就緒"""
    import subprocess
    import time
    import requests
    
    def is_ollama_ready():
        """檢查 Ollama 服務是否已準備就緒"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except (requests.RequestException, ConnectionError):
            return False
    
    try:
        # 先檢查是否已經在運行
        if is_ollama_ready():
            logger.info("Ollama 服務已經在運行中")
            return True
            
        logger.info("正在啟動 Ollama 服務...")
        
        # 使用 nohup 在後台啟動 Ollama
        subprocess.Popen(
            ["nohup", "ollama", "serve"],
            stdout=open("ollama.log", "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        
        # 等待服務啟動，最多等待 30 秒
        max_wait_time = 30  # 秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            if is_ollama_ready():
                logger.info("Ollama 服務已啟動並準備就緒")
                return True
            logger.info("等待 Ollama 服務啟動...")
            time.sleep(2)
            
        logger.error("等待 Ollama 服務啟動超時")
        return False
        
    except Exception as e:
        logger.error(f"啟動 Ollama 服務時出錯: {e}")
        return False

def get_available_models():
    """
    獲取本機可用的 Ollama 模型列表
    返回模型名稱列表
    """
    import requests
    import time
    
    max_retries = 5
    retry_delay = 3  # 秒
    
    for attempt in range(max_retries):
        try:
            # 使用 API 獲取模型列表
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            response.raise_for_status()
            
            # 解析 JSON 響應
            data = response.json()
            
            if 'models' not in data or not data['models']:
                logger.warning("未找到任何已下載的 Ollama 模型")
                return []
            
            # 提取模型名稱（去除 :latest 後綴）
            model_names = [model['name'].split(':')[0] for model in data['models']]
            
            # 去重並排序
            model_names = sorted(list(set(model_names)))
            
            if not model_names:
                logger.warning("未找到任何已下載的 Ollama 模型")
                return []
                
            logger.info(f"成功獲取到 {len(model_names)} 個模型")
            return model_names
            
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.warning(f"獲取模型列表失敗，{wait_time}秒後重試 ({attempt + 1}/{max_retries})... 錯誤: {str(e)}")
                time.sleep(wait_time)
                continue
            logger.error(f"獲取模型列表時出錯: {e}")
            return []
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.warning(f"處理模型列表時出錯，{wait_time}秒後重試 ({attempt + 1}/{max_retries})... 錯誤: {str(e)}")
                time.sleep(wait_time)
                continue
            logger.error(f"處理模型列表時出錯: {e}")
            return []

def load_prompt_templates():
    """載入專案中的提示詞模板"""
    import os
    from pathlib import Path
    
    # 確保路徑是絕對路徑
    project_root = Path(__file__).parent
    
    # 載入最佳提示詞模板
    best_prompt_path = project_root / "prompt_optimization_results" / "best_prompt_20250522_134414.txt"
    prompt_templates = []
    
    if best_prompt_path.exists():
        try:
            with open(best_prompt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # 確保檔案不是空的
                    prompt_templates.append(content)
                    logger.info(f"已載入提示詞模板: {best_prompt_path}")
                else:
                    logger.warning(f"提示詞模板檔案為空: {best_prompt_path}")
        except Exception as e:
            logger.error(f"載入提示詞模板時出錯: {e}")
    else:
        logger.warning(f"找不到提示詞模板檔案: {best_prompt_path}")
    
    # 如果沒有找到模板，使用預設模板
    if not prompt_templates:
        logger.info("使用預設提示詞模板")
        prompt_templates = ["""
        你是會議記錄整理助手。請將下面的逐字稿整理成清晰的會議記錄。

        核心任務：
        - 提取討論重點，不要逐字複製
        - 標記重要決定和行動事項
        - 用自己的話概括發言內容

        整理原則：
        - 保持客觀中立
        - 語言簡潔明瞭
        - 結構條理分明

        逐字稿：
        {transcript}

        請整理成包含以下部分的會議記錄：
        1. 會議概況
        2. 主要討論
        3. 決議事項
        4. 待辦任務
        """]
    
    logger.info(f"總共載入 {len(prompt_templates)} 個提示詞模板")
    return prompt_templates

def parse_arguments():
    """解析命令行參數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='會議記錄優化工具')
    parser.add_argument('--model', type=str, help='要使用的模型名稱')
    parser.add_argument('--iterations', type=int, default=10, help='優化迭代次數')
    return parser.parse_args()

def main():
    """主函數"""
    args = parse_arguments()
    logger.info(f"命令行參數: {vars(args)}")
    
    # 判斷是否自動批次模式
    auto_mode = False
    max_iterations = None
    min_improvement = None
    if hasattr(args, 'iterations') and args.iterations:
        auto_mode = True
        max_iterations = args.iterations
        min_improvement = 0.01

    # 只保留主邏輯呼叫
    run_main_logic(args, auto_mode=auto_mode, max_iterations=max_iterations, min_improvement=min_improvement)


def run_main_logic(args=None, auto_mode=False, max_iterations=None, min_improvement=None):
    """執行程式的主要邏輯"""
    if args is None:
        args = type('Args', (), {'model': None, 'iterations': 10})()
        auto_mode = False
    # 強制覆蓋互動 input
    if auto_mode and max_iterations is not None and min_improvement is not None:
        pass # 使用傳入參數
    # 以下為原本的主流程，確保只有一個 try 區塊
    try:
        # 載入測試案例
        test_cases = load_test_cases()
        if not test_cases:
            logger.error("沒有找到任何測試案例，程式結束。")
            exit(1)

        # 載入提示詞模板
        prompt_templates = load_prompt_templates()
        logger.info(f"已載入 {len(prompt_templates)} 個提示詞模板")
        
        # 獲取可用的 Ollama 模型
        # ...（其餘原本邏輯不變）

        prompt_templates = load_prompt_templates()
        logger.info(f"已載入 {len(prompt_templates)} 個提示詞模板")
        
        # 獲取可用的 Ollama 模型
        available_models = get_available_models()
        
        # 如果沒有可用的模型，提示用戶並退出
        if not available_models:
            logger.error("未找到可用的 Ollama 模型。請先使用 'ollama pull <model_name>' 下載模型。")
            exit(1)
        
        # 如果通過命令行指定了模型，則使用該模型
        if args and args.model:
            model_name = args.model
            logger.info(f"使用命令行指定的模型: {model_name}")
        elif auto_mode:
            # 自動模式下直接選第一個模型
            model_name = available_models[0]
            logger.info(f"自動模式下使用預設模型: {model_name}")
        else:
            # 顯示可用的模型並讓用戶選擇
            print("\n可用的 Ollama 模型：")
            for i, model in enumerate(available_models, 1):
                print(f"{i}. {model}")
            while True:
                try:
                    choice = input(f"\n請選擇要使用的模型 (1-{len(available_models)}) 或輸入模型名稱: ").strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                        model_name = available_models[int(choice) - 1]
                        break
                    elif choice in available_models:
                        model_name = choice
                        break
                    elif not choice and available_models:
                        model_name = available_models[0]
                        logger.info(f"使用預設模型: {model_name}")
                        break
                    else:
                        print(f"無效的選擇，請輸入 1-{len(available_models)} 之間的數字或有效的模型名稱。")
                except (ValueError, IndexError):
                    print(f"無效的輸入，請輸入 1-{len(available_models)} 之間的數字或有效的模型名稱。")
        
        logger.info(f"初始化會議記錄生成器，使用模型: {model_name}")
        miner = MeetingMiner(model_name=model_name)
        
        # 進行優化
        if auto_mode and max_iterations is not None:
            _max_iterations = max_iterations
        else:
            _max_iterations = int(input("請輸入最大迭代次數 (預設: 10): ") or 10)
        
        for test_case in test_cases:
            logger.info(f"\n=== 開始優化：{test_case['transcript_file']} ===")
            best_score = -1
            best_prompt = ""
            best_result = ""
            
            for iteration in range(_max_iterations):
                # 使用不同的提示詞模板
                prompt = prompt_templates[iteration % len(prompt_templates)]
                exec_time = None
                try:
                    import time
                    start_time = time.time()
                    # 產生會議紀錄
                    logger.info(f"使用提示詞 {iteration + 1}/{_max_iterations} 生成會議紀錄...")
                    result = miner.generate_minutes(test_case["transcript"], prompt)
                    end_time = time.time()
                    exec_time = end_time - start_time
                    logger.info(f"模型: {model_name}, 測試案例: {test_case['transcript_file']}, 迭代: {iteration+1}, 執行時間: {exec_time:.2f} 秒")
                    # 評估結果
                    if test_case.get("reference"):
                        eval_result = evaluate(result["content"], test_case["reference"])
                        score = eval_result["score"]
                        logger.info(f"第 {iteration + 1} 次迭代 - 分數: {score:.4f}")
                    else:
                        score = 0
                        logger.warning(f"第 {iteration + 1} 次迭代 - 無參考資料，無法評分")
                    # 記錄每次執行的結果（含 prompt, score, exec_time, content...）
                    result_key = f"{model_name}__{test_case['transcript_file']}__iter{iteration+1}"
                    all_result = {
                        'model': model_name,
                        'test_case': test_case['transcript_file'],
                        'iteration': iteration+1,
                        'prompt': prompt,
                        'score': score,
                        'exec_time_sec': exec_time,
                        'content': result["content"]
                    }
                    if test_case.get("reference"):
                        all_result['reference'] = test_case['reference']
                    if eval_result:
                        all_result.update(eval_result)
                    if 'all_results' not in locals():
                        all_results = {}
                    all_results[result_key] = all_result
                    # 更新最佳結果
                    if score > best_score:
                        best_score = score
                        best_prompt = prompt
                        best_result = result["content"]
                except Exception as e:
                    logger.error(f"第 {iteration + 1} 次迭代出錯: {e}", exc_info=True)
                    continue
            
            # 輸出最佳結果
            logger.info(f"\n{'='*40}")
            logger.info(f"檔案: {test_case['transcript_file']}")
            logger.info(f"最佳分數: {best_score:.4f}")
            
            # 儲存結果
            output_dir = Path("optimization_results")
            output_dir.mkdir(exist_ok=True)
            
            # 安全處理檔案名稱
            safe_filename = "".join(c if c.isalnum() or c in ' ._-' else '_' for c in test_case['transcript_file'])
            safe_model_name = "".join(c if c.isalnum() or c in ' ._-' else '_' for c in model_name) # model_name is available in this scope
            output_file = output_dir / f"best_{safe_model_name}_{safe_filename}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"檔案: {test_case['transcript_file']}\n")
                f.write(f"分數: {best_score:.4f}\n\n")
                f.write("提示詞:\n")
                f.write(best_prompt)
                f.write("\n\n生成的會議紀錄:\n")
                f.write(best_result)
            
            logger.info(f"結果已儲存至: {output_file}")
            
            # 顯示最佳結果摘要
            logger.info("\n最佳提示詞 (摘要):")
            logger.info(best_prompt[:200] + ("..." if len(best_prompt) > 200 else ""))
            
            logger.info("\n最佳生成會議紀錄 (摘要):")
            logger.info(best_result[:500] + ("..." if len(best_result) > 500 else ""))
            logger.info('='*40)
        
            logger.info("\n所有優化完成！")
        
    except KeyboardInterrupt:
        logger.info("\n使用者中斷，程式結束。")
    except Exception as e:
        logger.error(f"程式執行出錯: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
