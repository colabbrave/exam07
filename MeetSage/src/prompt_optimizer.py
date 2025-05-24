class PromptOptimizer:
    """提示詞優化器類別"""

    def __init__(self, evaluator=None, meeting_miner=None, test_cases=None, model_name=None):
        self.evaluator = evaluator
        self.meeting_miner = meeting_miner
        self.test_cases = test_cases
        self.model_name = model_name

    def optimize(self, max_iterations=10, min_improvement=0.01):
        """執行優化邏輯：只用一個專業深度助理提示詞，產生一組結果"""
        if not self.test_cases or not self.meeting_miner or not self.evaluator:
            return {
                "all_results": {},
                "best_variant": None,
                "best_score": 0.0
            }
        prompt = (
            "你是專業的深度會議記錄助理，你的任務是把逐字稿轉化為一篇能呈現發言精髓、討論脈絡和決策要點的會議紀錄。請在生成過程中保持以下原則：\n"
            "\n抓住核心：對每段發言，用自己的語言去重述討論焦點，不要原文搬運。\n"
            "\n挖掘邏輯：將發言中隱含的因果、爭議或建議，化為條理清晰的段落，並標明各方立場或結論。\n"
            "\n精簡語言：摒除多餘形容詞與副詞，儘量以名詞和動詞呈現信息；對於核心概念與決策，請以 粗體 標示。\n"
            "\n結構分明：開頭概述會議目的與整體氛圍；中段按主題或議題分段；結尾歸納行動項目與後續安排。\n"
            "\n語調沉穩：像思想家一樣嚴謹，語句自然流暢，不額外提問或打斷讀者思路。\n"
            "\n在這些指引下，請將逐字稿轉成一篇既專業又富洞見的會議紀錄。"
        )
        all_results = {}
        best_score = 0.0
        best_variant = None
        total_score = 0.0
        for idx, case in enumerate(self.test_cases):
            result = self.meeting_miner.generate_minutes(
                transcript=case['transcript'],
                prompt_template=prompt
            )
            score_obj = self.evaluator.evaluate(case['reference'], result['content'])
            score = score_obj.avg_score
            total_score += score
            all_results[f"professional_assistant_{idx}"] = {
                "metrics": {"avg_score": score},
                "description": "專業深度會議記錄助理指引",
                "template": prompt
            }
            if score > best_score:
                best_score = score
                best_variant = prompt
        avg_score = total_score / len(self.test_cases)
        return {
            "all_results": all_results,
            "best_variant": best_variant,
            "best_score": best_score,
            "avg_score": avg_score
        }