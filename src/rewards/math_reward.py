import re
import torch
import numpy as np
from math_verify import parse, verify
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass


def correctness_reward_func(prompts, completions, solution, completion_ids, **kwargs) -> list[float]:
    """Reward function that checks if the response matches the correct answer. (For training)

    Args:
        prompts: Input prompts
        completions: Model-generated responses
        solution: Correct answers
        completion_ids: Generated token IDs
        **kwargs: Additional arguments

    Returns:
        list[float]: Reward value for each sample (1.0 for correct, 0.0 for incorrect)
    """
    responses = [completion[0]['content'] for completion in completions]
    solutions = solution

    extracted_responses = [parse(r) for r in responses]
    extracted_solutions = [parse(a) for a in solutions]
    is_correct_list = [1.0 if verify(r, a) else 0.0 for r, a in zip(extracted_responses, extracted_solutions)]

    return is_correct_list


# ============ Helper Functions for Quality Evaluation ============

def evaluate_reasoning_quality(completion_text: str, min_steps: int = 2, max_steps: int = 8) -> float:
    """
    评估推理质量

    Args:
        completion_text: 生成的推理文本
        min_steps: 最少推理步数
        max_steps: 最多推理步数（超过视为冗余）

    Returns:
        quality_score: 0-1 之间的质量分数
    """
    if not completion_text or len(completion_text) < 20:
        return 0.0

    score = 0.0

    # 1. 是否有 boxed 答案 (0.25)
    if 'boxed' in completion_text:
        score += 0.25

    # 2. 推理步骤数量 (0.25)
    step_separators = ['\n\n', '\n- ', '\n* ', '\n1.', '\n2.', '\n3.', '\n4.', '\n5.']
    max_steps_count = 0
    for sep in step_separators:
        count = completion_text.count(sep)
        if count > max_steps_count:
            max_steps_count = count

    steps = max_steps_count + 1
    if min_steps <= steps <= max_steps:
        score += 0.25
    elif steps > max_steps:
        score += 0.1

    # 3. 是否有数学表达式或计算 (0.25)
    math_patterns = [
        r'[\+\-\*\/=]', r'\$', r'\\frac', r'\\times', r'\\sqrt',
        r'\\sum', r'\\prod', r'\\int', r'\\lim', r'\\log',
        r'\\sin', r'\\cos', r'\\tan', r'\\pi', r'\\infty'
    ]
    for pattern in math_patterns:
        if re.search(pattern, completion_text):
            score += 0.25
            break

    # 4. 逻辑连接词 (0.125)
    logic_words = [
        'because', 'therefore', 'thus', 'since', 'hence',
        'so', 'then', 'consequently', 'as a result',
        'we have', 'we get', 'we find', 'we obtain',
        'let', 'assume', 'suppose', 'consider'
    ]
    logic_count = sum(1 for word in logic_words if word in completion_text.lower())
    if logic_count >= 2:
        score += 0.125
    elif logic_count >= 1:
        score += 0.075

    # 5. 长度合理性 (0.125)
    length = len(completion_text)
    if 100 <= length <= 800:
        score += 0.125
    elif 50 <= length < 100:
        score += 0.075

    return min(score, 1.0)


def is_high_quality(completion_text: str, quality_threshold: float = 0.6) -> bool:
    """判断是否为高质量推理"""
    if not completion_text or len(completion_text) < 30:
        return False
    quality_score = evaluate_reasoning_quality(completion_text)
    return quality_score >= quality_threshold


def compute_avg_confidence(logits: torch.Tensor) -> torch.Tensor:
    """计算每个生成序列的平均置信度"""
    if logits.dim() == 2:
        probs = torch.softmax(logits, dim=-1)
        confidence, _ = probs.max(dim=-1)
        return confidence
    elif logits.dim() == 3:
        probs = torch.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)
        confidence = max_probs.mean(dim=-1)
        return confidence
    else:
        raise ValueError(f"Unsupported logits shape: {logits.shape}")


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """计算每个生成序列的平均熵"""
    if logits.dim() == 2:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return entropy
    elif logits.dim() == 3:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        entropy = entropy.mean(dim=-1)
        return entropy
    else:
        raise ValueError(f"Unsupported logits shape: {logits.shape}")


# ============ 1. RelativeLengthRewarder (LEPO paper) ============

@dataclass
class RelativeLengthRewarder:
    """
    Implements the relative length penalty/reward mechanism from the LEPO paper.
    """
    num_generations: int
    length_judge: callable
    answer_reward: float = 1.0
    format_penalty: float = -1.0
    relative_length_penalty: float = 1.0
    relative_length_reward: Optional[float] = "same"
    accuracy_requirement: float = 0.5
    all_gather: bool = False

    __name__: str = "RelativeLengthRewarder"

    def __post_init__(self):
        if self.format_penalty > 0:
            raise ValueError(f"format_penalty must be <= 0. Got: {self.format_penalty}")
        if self.accuracy_requirement < 0 or self.accuracy_requirement > 1:
            raise ValueError(f"accuracy_requirement must be between 0 and 1. Got: {self.accuracy_requirement}")

    def __call__(
            self,
            prompts: List[str],
            completions: List[Any],
            completion_ids: List[List[int]],
            prompt_ids: List[int] = None,
            solutions: List[Any] = None,
            **kwargs
    ) -> List[float]:
        """Calculate rewards with relative length adjustment."""
        completions_text = self._extract_completion_text(completions)

        if solutions is None:
            solutions = kwargs.get('solution', kwargs.get('label', [None] * len(prompts)))
        if not isinstance(solutions, list):
            solutions = [solutions] * len(prompts)
        if len(solutions) < len(prompts):
            solutions = solutions + [None] * (len(prompts) - len(solutions))
        if prompt_ids is None:
            prompt_ids = [i // self.num_generations for i in range(len(prompts))]

        format_map = self._check_format(completions_text)
        correct_map = self._check_correctness(completions_text, solutions, format_map)
        length_map = self._compute_lengths(completion_ids, correct_map)
        rel_length_adjustments = self._compute_relative_length_adjustments(
            prompt_ids, length_map, correct_map
        )
        rewards = self._combine_rewards(format_map, correct_map, rel_length_adjustments)
        return rewards

    def _extract_completion_text(self, completions):
        completions_text = []
        for comp in completions:
            if comp is None:
                completions_text.append("")
            elif isinstance(comp, str):
                completions_text.append(comp)
            elif isinstance(comp, list) and len(comp) > 0:
                if isinstance(comp[0], dict):
                    if 'content' in comp[0]:
                        completions_text.append(comp[0]['content'])
                    elif 'text' in comp[0]:
                        completions_text.append(comp[0]['text'])
                    else:
                        completions_text.append(str(comp[0]))
                else:
                    completions_text.append(str(comp[0]))
            elif isinstance(comp, dict):
                if 'content' in comp:
                    completions_text.append(comp['content'])
                elif 'text' in comp:
                    completions_text.append(comp['text'])
                else:
                    completions_text.append(str(comp))
            else:
                completions_text.append(str(comp))
        return completions_text

    def _check_format(self, completions):
        format_map = []
        for completion in completions:
            if completion is None or not isinstance(completion, str):
                is_formatted = False
            else:
                is_formatted = bool(re.search(r'\\boxed\{.*?\}', completion))
            format_map.append(is_formatted)
        return format_map

    def _check_correctness(self, completions, solutions, format_map):
        correct_map = []
        for completion, solution, is_formatted in zip(completions, solutions, format_map):
            if not is_formatted or solution is None or completion is None:
                correct_map.append(False)
            else:
                try:
                    extracted_response = parse(completion)
                    extracted_solution = parse(solution)
                    correct_map.append(verify(extracted_response, extracted_solution))
                except:
                    correct_map.append(False)
        return correct_map

    def _compute_lengths(self, completion_ids, correct_map):
        length_map = []
        for is_correct, cid in zip(correct_map, completion_ids):
            if is_correct:
                try:
                    length = self.length_judge(cid)
                    length_map.append(float(length))
                except:
                    length_map.append(float('nan'))
            else:
                length_map.append(float('nan'))
        return length_map

    def _compute_relative_length_adjustments(self, prompt_ids, length_map, correct_map):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        length_tensor = torch.tensor(length_map, dtype=torch.float32, device=device)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        correct_tensor = torch.tensor(correct_map, dtype=torch.bool, device=device)
        rel_length_adjustments = torch.zeros_like(length_tensor)
        unique_prompt_ids = set(prompt_ids)

        for pid in unique_prompt_ids:
            prompt_mask = prompt_tensor == pid
            group_size = prompt_mask.sum().item()
            if group_size < self.num_generations:
                continue
            valid_mask = prompt_mask & ~length_tensor.isnan()
            valid_lengths = length_tensor[valid_mask]
            if len(valid_lengths) == 0:
                continue
            n_correct = correct_tensor[prompt_mask].sum().item()
            group_accuracy = n_correct / group_size
            if group_accuracy >= self.accuracy_requirement:
                rel_len_coeff = -self.relative_length_penalty
            else:
                if self.relative_length_reward == "same":
                    rel_len_coeff = self.relative_length_penalty
                else:
                    rel_len_coeff = self.relative_length_reward or 0.0
            max_len = valid_lengths.max().item()
            min_len = valid_lengths.min().item()
            mean_len = valid_lengths.mean().item()
            len_range = max_len - min_len
            if len_range > 0:
                local_valid_mask = (prompt_tensor == pid) & valid_mask
                normalized_lengths = (length_tensor[local_valid_mask] - mean_len) / len_range
                rel_length_adjustments[local_valid_mask] = rel_len_coeff * normalized_lengths
        return rel_length_adjustments.tolist()

    def _combine_rewards(self, format_map, correct_map, adjustments):
        rewards = []
        for is_formatted, is_correct, adj in zip(format_map, correct_map, adjustments):
            reward = 0.0
            if not is_formatted:
                reward += self.format_penalty
            if is_correct:
                reward += self.answer_reward
            reward += adj
            rewards.append(reward)
        return rewards


# ============ 2. ContrastiveLengthRewarder (Idea 3 - 方案A) ============

@dataclass
class ContrastiveLengthRewarder:
    """
    对比长度优化奖励器 - 方案A: 只对正确样本应用
    """
    num_generations: int
    length_judge: callable
    answer_reward: float = 1.0
    format_penalty: float = -0.3
    contrastive_coefficient: float = 0.3
    top_k_short: int = 1
    length_ratio_threshold: float = 1.5
    temperature: float = 0.3
    min_group_accuracy: float = 0.3
    min_correct_samples: int = 2
    use_absolute_diff: bool = True
    min_length_diff: float = 30.0

    __name__: str = "ContrastiveLengthRewarder"

    def __post_init__(self):
        if self.format_penalty > 0:
            raise ValueError(f"format_penalty must be <= 0. Got: {self.format_penalty}")
        if self.top_k_short < 1:
            raise ValueError(f"top_k_short must be >= 1. Got: {self.top_k_short}")

    def __call__(self, prompts, completions, completion_ids, prompt_ids=None, solutions=None, **kwargs):
        completions_text = self._extract_completion_text(completions)
        if solutions is None:
            solutions = kwargs.get('solution', kwargs.get('label', [None] * len(prompts)))
        if not isinstance(solutions, list):
            solutions = [solutions] * len(prompts)
        if len(solutions) < len(prompts):
            solutions = solutions + [None] * (len(prompts) - len(solutions))
        if prompt_ids is None:
            prompt_ids = [i // self.num_generations for i in range(len(prompts))]

        format_map = self._check_format(completions_text)
        correct_map = self._check_correctness(completions_text, solutions, format_map)
        length_map = self._compute_lengths(completion_ids, correct_map)
        contrastive_adjustments = self._compute_contrastive_adjustments(
            prompt_ids, length_map, correct_map
        )
        return self._combine_rewards(format_map, correct_map, contrastive_adjustments)

    def _extract_completion_text(self, completions):
        completions_text = []
        for comp in completions:
            if comp is None:
                completions_text.append("")
            elif isinstance(comp, str):
                completions_text.append(comp)
            elif isinstance(comp, list) and len(comp) > 0:
                if isinstance(comp[0], dict):
                    if 'content' in comp[0]:
                        completions_text.append(comp[0]['content'])
                    elif 'text' in comp[0]:
                        completions_text.append(comp[0]['text'])
                    else:
                        completions_text.append(str(comp[0]))
                else:
                    completions_text.append(str(comp[0]))
            elif isinstance(comp, dict):
                if 'content' in comp:
                    completions_text.append(comp['content'])
                elif 'text' in comp:
                    completions_text.append(comp['text'])
                else:
                    completions_text.append(str(comp))
            else:
                completions_text.append(str(comp))
        return completions_text

    def _check_format(self, completions):
        format_map = []
        for completion in completions:
            if completion is None or not isinstance(completion, str):
                is_formatted = False
            else:
                is_formatted = bool(re.search(r'\\boxed\{.*?\}', completion))
            format_map.append(is_formatted)
        return format_map

    def _check_correctness(self, completions, solutions, format_map):
        correct_map = []
        for completion, solution, is_formatted in zip(completions, solutions, format_map):
            if not is_formatted or solution is None or completion is None:
                correct_map.append(False)
            else:
                try:
                    extracted_response = parse(completion)
                    extracted_solution = parse(solution)
                    correct_map.append(verify(extracted_response, extracted_solution))
                except:
                    correct_map.append(False)
        return correct_map

    def _compute_lengths(self, completion_ids, correct_map):
        length_map = []
        for is_correct, cid in zip(correct_map, completion_ids):
            if is_correct:
                try:
                    length = self.length_judge(cid)
                    length_map.append(float(length))
                except:
                    length_map.append(float('nan'))
            else:
                length_map.append(float('nan'))
        return length_map

    def _compute_contrastive_adjustments(self, prompt_ids, length_map, correct_map):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        length_tensor = torch.tensor(length_map, dtype=torch.float32, device=device)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        correct_tensor = torch.tensor(correct_map, dtype=torch.bool, device=device)
        contrastive_adjustments = torch.zeros_like(length_tensor)

        total_groups = 0
        contrast_applied = 0
        unique_prompt_ids = set(prompt_ids)

        for pid in unique_prompt_ids:
            prompt_mask = prompt_tensor == pid
            group_size = prompt_mask.sum().item()
            if group_size < self.num_generations:
                continue
            total_groups += 1

            correct_mask = prompt_mask & correct_tensor
            correct_lengths = length_tensor[correct_mask]
            n_correct = correct_tensor[prompt_mask].sum().item()
            group_accuracy = n_correct / group_size

            if group_accuracy < self.min_group_accuracy:
                continue
            if len(correct_lengths) < self.min_correct_samples:
                continue

            sorted_lengths, sorted_indices = torch.sort(correct_lengths)
            n_short = min(self.top_k_short, len(sorted_lengths))
            n_long = min(self.top_k_short, len(sorted_lengths))
            short_lengths = sorted_lengths[:n_short]
            long_lengths = sorted_lengths[-n_long:]
            short_avg = short_lengths.mean().item()
            long_avg = long_lengths.mean().item()

            length_diff = long_avg - short_avg
            should_apply = False
            if self.use_absolute_diff and length_diff > self.min_length_diff:
                should_apply = True
            if long_avg > short_avg * self.length_ratio_threshold:
                should_apply = True

            if should_apply:
                contrast_applied += 1
                all_correct_indices = torch.where(correct_mask)[0]
                short_indices = all_correct_indices[sorted_indices[:n_short]]
                long_indices = all_correct_indices[sorted_indices[-n_long:]]
                norm_diff = min(length_diff / max(long_avg, 1.0), 1.0)
                reward_val = self.contrastive_coefficient * norm_diff
                contrastive_adjustments[short_indices] += reward_val
                contrastive_adjustments[long_indices] -= reward_val

        return contrastive_adjustments.tolist()

    def _combine_rewards(self, format_map, correct_map, adjustments):
        rewards = []
        for is_formatted, is_correct, adj in zip(format_map, correct_map, adjustments):
            reward = 0.0
            if not is_formatted:
                reward += self.format_penalty
            if is_correct:
                reward += self.answer_reward
            reward += adj
            rewards.append(reward)
        return rewards


# ============ 3. HybridContrastiveLengthRewarder ============

@dataclass
class HybridContrastiveLengthRewarder:
    """
    混合对比长度奖励器
    """
    num_generations: int
    length_judge: callable
    answer_reward: float = 1.0
    format_penalty: float = -1.0
    relative_length_penalty: float = 1.0
    accuracy_requirement: float = 0.5
    contrastive_coefficient: float = 0.5
    top_k_short: int = 2
    length_ratio_threshold: float = 1.5
    use_contrastive_for_hard: bool = True
    min_length_diff: float = 20.0
    length_ratio_threshold_abs: float = 1.3
    min_group_accuracy: float = 0.2

    __name__: str = "HybridContrastiveLengthRewarder"

    def __post_init__(self):
        if self.format_penalty > 0:
            raise ValueError(f"format_penalty must be <= 0. Got: {self.format_penalty}")

    def __call__(self, prompts, completions, completion_ids, prompt_ids=None, solutions=None, **kwargs):
        completions_text = self._extract_completion_text(completions)
        if solutions is None:
            solutions = kwargs.get('solution', kwargs.get('label', [None] * len(prompts)))
        if not isinstance(solutions, list):
            solutions = [solutions] * len(prompts)
        if len(solutions) < len(prompts):
            solutions = solutions + [None] * (len(prompts) - len(solutions))
        if prompt_ids is None:
            prompt_ids = [i // self.num_generations for i in range(len(prompts))]

        format_map = self._check_format(completions_text)
        correct_map = self._check_correctness(completions_text, solutions, format_map)
        length_map = self._compute_lengths(completion_ids, correct_map)
        rel_adjustments = self._compute_relative_adjustments(prompt_ids, length_map, correct_map)
        contrast_adjustments = self._compute_contrastive_adjustments(prompt_ids, length_map, correct_map)
        final_adjustments = self._merge_adjustments(prompt_ids, correct_map, rel_adjustments, contrast_adjustments)
        return self._combine_rewards(format_map, correct_map, final_adjustments)

    def _extract_completion_text(self, completions):
        completions_text = []
        for comp in completions:
            if comp is None:
                completions_text.append("")
            elif isinstance(comp, str):
                completions_text.append(comp)
            elif isinstance(comp, list) and len(comp) > 0:
                if isinstance(comp[0], dict):
                    if 'content' in comp[0]:
                        completions_text.append(comp[0]['content'])
                    elif 'text' in comp[0]:
                        completions_text.append(comp[0]['text'])
                    else:
                        completions_text.append(str(comp[0]))
                else:
                    completions_text.append(str(comp[0]))
            elif isinstance(comp, dict):
                if 'content' in comp:
                    completions_text.append(comp['content'])
                elif 'text' in comp:
                    completions_text.append(comp['text'])
                else:
                    completions_text.append(str(comp))
            else:
                completions_text.append(str(comp))
        return completions_text

    def _check_format(self, completions):
        format_map = []
        for completion in completions:
            if completion is None or not isinstance(completion, str):
                is_formatted = False
            else:
                is_formatted = bool(re.search(r'\\boxed\{.*?\}', completion))
            format_map.append(is_formatted)
        return format_map

    def _check_correctness(self, completions, solutions, format_map):
        correct_map = []
        for completion, solution, is_formatted in zip(completions, solutions, format_map):
            if not is_formatted or solution is None or completion is None:
                correct_map.append(False)
            else:
                try:
                    extracted_response = parse(completion)
                    extracted_solution = parse(solution)
                    correct_map.append(verify(extracted_response, extracted_solution))
                except:
                    correct_map.append(False)
        return correct_map

    def _compute_lengths(self, completion_ids, correct_map):
        length_map = []
        for is_correct, cid in zip(correct_map, completion_ids):
            if is_correct:
                try:
                    length = self.length_judge(cid)
                    length_map.append(float(length))
                except:
                    length_map.append(float('nan'))
            else:
                length_map.append(float('nan'))
        return length_map

    def _compute_relative_adjustments(self, prompt_ids, length_map, correct_map):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        length_tensor = torch.tensor(length_map, dtype=torch.float32, device=device)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        correct_tensor = torch.tensor(correct_map, dtype=torch.bool, device=device)
        adjustments = torch.zeros_like(length_tensor)
        unique_prompt_ids = set(prompt_ids)

        for pid in unique_prompt_ids:
            prompt_mask = prompt_tensor == pid
            group_size = prompt_mask.sum().item()
            if group_size < self.num_generations:
                continue
            valid_mask = prompt_mask & ~length_tensor.isnan()
            valid_lengths = length_tensor[valid_mask]
            if len(valid_lengths) == 0:
                continue
            n_correct = correct_tensor[prompt_mask].sum().item()
            group_accuracy = n_correct / group_size
            if group_accuracy >= self.accuracy_requirement:
                max_len = valid_lengths.max().item()
                min_len = valid_lengths.min().item()
                mean_len = valid_lengths.mean().item()
                len_range = max_len - min_len
                if len_range > 0:
                    local_valid_mask = (prompt_tensor == pid) & valid_mask
                    normalized_lengths = (length_tensor[local_valid_mask] - mean_len) / len_range
                    adjustments[local_valid_mask] = -self.relative_length_penalty * normalized_lengths
        return adjustments.tolist()

    def _compute_contrastive_adjustments(self, prompt_ids, length_map, correct_map):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        length_tensor = torch.tensor(length_map, dtype=torch.float32, device=device)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        correct_tensor = torch.tensor(correct_map, dtype=torch.bool, device=device)
        adjustments = torch.zeros_like(length_tensor)
        unique_prompt_ids = set(prompt_ids)

        for pid in unique_prompt_ids:
            prompt_mask = prompt_tensor == pid
            group_size = prompt_mask.sum().item()
            if group_size < self.num_generations:
                continue
            correct_mask = prompt_mask & correct_tensor
            correct_lengths = length_tensor[correct_mask]
            n_correct = correct_tensor[prompt_mask].sum().item()
            group_accuracy = n_correct / group_size
            if group_accuracy >= self.accuracy_requirement:
                continue
            if group_accuracy < self.min_group_accuracy:
                continue
            if len(correct_lengths) < 2:
                continue
            sorted_lengths, sorted_indices = torch.sort(correct_lengths)
            n_short = min(self.top_k_short, len(sorted_lengths))
            n_long = min(self.top_k_short, len(sorted_lengths))
            short_lengths = sorted_lengths[:n_short]
            long_lengths = sorted_lengths[-n_long:]
            short_avg = short_lengths.mean().item()
            long_avg = long_lengths.mean().item()
            length_diff = long_avg - short_avg
            if length_diff > self.min_length_diff or long_avg > short_avg * self.length_ratio_threshold_abs:
                all_correct_indices = torch.where(correct_mask)[0]
                short_indices = all_correct_indices[sorted_indices[:n_short]]
                long_indices = all_correct_indices[sorted_indices[-n_long:]]
                norm_diff = min(length_diff / max(long_avg, 1.0), 1.0)
                reward_val = self.contrastive_coefficient * norm_diff
                adjustments[short_indices] += reward_val
                adjustments[long_indices] -= reward_val
        return adjustments.tolist()

    def _merge_adjustments(self, prompt_ids, correct_map, rel_adjustments, contrast_adjustments):
        final = []
        for rel, cont, is_correct in zip(rel_adjustments, contrast_adjustments, correct_map):
            if not is_correct:
                final.append(0.0)
            else:
                final.append(rel + cont)
        return final

    def _combine_rewards(self, format_map, correct_map, adjustments):
        rewards = []
        for is_formatted, is_correct, adj in zip(format_map, correct_map, adjustments):
            reward = 0.0
            if not is_formatted:
                reward += self.format_penalty
            if is_correct:
                reward += self.answer_reward
            reward += adj
            rewards.append(reward)
        return rewards


# ============ 4. QualityFilteredContrastiveRewarder (思路2) ============

@dataclass
class QualityFilteredContrastiveRewarder:
    """
    质量筛选对比奖励器 (思路2)
    """
    num_generations: int
    length_judge: callable
    answer_reward: float = 1.0
    format_penalty: float = -0.3
    quality_threshold: float = 0.6
    contrastive_coefficient: float = 0.2
    top_k_short: int = 1
    length_ratio_threshold: float = 1.5
    min_correct_samples: int = 2
    min_high_quality_samples: int = 2
    min_length_diff: float = 30.0
    min_steps: int = 2
    max_steps: int = 8

    __name__: str = "QualityFilteredContrastiveRewarder"

    def __post_init__(self):
        if self.format_penalty > 0:
            raise ValueError(f"format_penalty must be <= 0. Got: {self.format_penalty}")

    def __call__(self, prompts, completions, completion_ids, prompt_ids=None, solutions=None, **kwargs):
        completions_text = self._extract_completion_text(completions)
        if solutions is None:
            solutions = kwargs.get('solution', kwargs.get('label', [None] * len(prompts)))
        if not isinstance(solutions, list):
            solutions = [solutions] * len(prompts)
        if len(solutions) < len(prompts):
            solutions = solutions + [None] * (len(prompts) - len(solutions))
        if prompt_ids is None:
            prompt_ids = [i // self.num_generations for i in range(len(prompts))]

        format_map = self._check_format(completions_text)
        correct_map = self._check_correctness(completions_text, solutions, format_map)
        length_map = self._compute_lengths(completion_ids, correct_map)
        quality_map = self._compute_quality_scores(completions_text, correct_map)
        contrastive_adjustments = self._compute_quality_filtered_contrastive(
            prompt_ids, length_map, correct_map, quality_map
        )
        return self._combine_rewards(format_map, correct_map, contrastive_adjustments)

    def _extract_completion_text(self, completions):
        completions_text = []
        for comp in completions:
            if comp is None:
                completions_text.append("")
            elif isinstance(comp, str):
                completions_text.append(comp)
            elif isinstance(comp, list) and len(comp) > 0:
                if isinstance(comp[0], dict):
                    if 'content' in comp[0]:
                        completions_text.append(comp[0]['content'])
                    elif 'text' in comp[0]:
                        completions_text.append(comp[0]['text'])
                    else:
                        completions_text.append(str(comp[0]))
                else:
                    completions_text.append(str(comp[0]))
            elif isinstance(comp, dict):
                if 'content' in comp:
                    completions_text.append(comp['content'])
                elif 'text' in comp:
                    completions_text.append(comp['text'])
                else:
                    completions_text.append(str(comp))
            else:
                completions_text.append(str(comp))
        return completions_text

    def _check_format(self, completions):
        format_map = []
        for completion in completions:
            if completion is None or not isinstance(completion, str):
                is_formatted = False
            else:
                is_formatted = bool(re.search(r'\\boxed\{.*?\}', completion))
            format_map.append(is_formatted)
        return format_map

    def _check_correctness(self, completions, solutions, format_map):
        correct_map = []
        for completion, solution, is_formatted in zip(completions, solutions, format_map):
            if not is_formatted or solution is None or completion is None:
                correct_map.append(False)
            else:
                try:
                    extracted_response = parse(completion)
                    extracted_solution = parse(solution)
                    correct_map.append(verify(extracted_response, extracted_solution))
                except:
                    correct_map.append(False)
        return correct_map

    def _compute_lengths(self, completion_ids, correct_map):
        length_map = []
        for is_correct, cid in zip(correct_map, completion_ids):
            if is_correct:
                try:
                    length = self.length_judge(cid)
                    length_map.append(float(length))
                except:
                    length_map.append(float('nan'))
            else:
                length_map.append(float('nan'))
        return length_map

    def _compute_quality_scores(self, completions_text, correct_map):
        quality_map = []
        for is_correct, comp_text in zip(correct_map, completions_text):
            if is_correct:
                quality = evaluate_reasoning_quality(comp_text, self.min_steps, self.max_steps)
                quality_map.append(quality)
            else:
                quality_map.append(0.0)
        return quality_map

    def _compute_quality_filtered_contrastive(self, prompt_ids, length_map, correct_map, quality_map):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        length_tensor = torch.tensor(length_map, dtype=torch.float32, device=device)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        correct_tensor = torch.tensor(correct_map, dtype=torch.bool, device=device)
        quality_tensor = torch.tensor(quality_map, dtype=torch.float32, device=device)
        contrastive_adjustments = torch.zeros_like(length_tensor)
        unique_prompt_ids = set(prompt_ids)

        for pid in unique_prompt_ids:
            prompt_mask = prompt_tensor == pid
            group_size = prompt_mask.sum().item()
            if group_size < self.num_generations:
                continue
            correct_mask = prompt_mask & correct_tensor
            n_correct = correct_tensor[prompt_mask].sum().item()
            if n_correct < self.min_correct_samples:
                continue
            high_quality_mask = correct_mask & (quality_tensor >= self.quality_threshold)
            high_quality_lengths = length_tensor[high_quality_mask]
            n_high_quality = high_quality_mask.sum().item()
            if n_high_quality < self.min_high_quality_samples:
                continue
            sorted_lengths, sorted_indices = torch.sort(high_quality_lengths)
            n_short = min(self.top_k_short, len(sorted_lengths))
            n_long = min(self.top_k_short, len(sorted_lengths))
            short_lengths = sorted_lengths[:n_short]
            long_lengths = sorted_lengths[-n_long:]
            short_avg = short_lengths.mean().item()
            long_avg = long_lengths.mean().item()
            length_diff = long_avg - short_avg
            if length_diff > self.min_length_diff or long_avg > short_avg * self.length_ratio_threshold:
                all_high_quality_indices = torch.where(high_quality_mask)[0]
                short_indices = all_high_quality_indices[sorted_indices[:n_short]]
                long_indices = all_high_quality_indices[sorted_indices[-n_long:]]
                norm_diff = min(length_diff / max(long_avg, 1.0), 1.0)
                reward_val = self.contrastive_coefficient * norm_diff
                contrastive_adjustments[short_indices] += reward_val
                contrastive_adjustments[long_indices] -= reward_val

        return contrastive_adjustments.tolist()

    def _combine_rewards(self, format_map, correct_map, adjustments):
        rewards = []
        for is_formatted, is_correct, adj in zip(format_map, correct_map, adjustments):
            reward = 0.0
            if not is_formatted:
                reward += self.format_penalty
            if is_correct:
                reward += self.answer_reward
            reward += adj
            rewards.append(reward)
        return rewards


# ============ 5. ConfidenceAwareLengthRewarder (思路5) ============

@dataclass
class ConfidenceAwareLengthRewarder:
    """
    置信度感知长度奖励器 (思路5)
    """
    num_generations: int
    length_judge: callable
    answer_reward: float = 1.0
    format_penalty: float = -0.3
    confidence_threshold: float = 0.7
    optimal_length_min: float = 150
    optimal_length_max: float = 350
    short_length_reward: float = 0.1
    optimal_length_reward: float = 0.15
    long_length_reward: float = 0.05
    use_entropy: bool = True
    entropy_threshold: float = 2.0

    __name__: str = "ConfidenceAwareLengthRewarder"

    def __post_init__(self):
        if self.format_penalty > 0:
            raise ValueError(f"format_penalty must be <= 0. Got: {self.format_penalty}")

    def __call__(self, prompts, completions, completion_ids, prompt_ids=None, solutions=None, logits=None, **kwargs):
        completions_text = self._extract_completion_text(completions)
        if solutions is None:
            solutions = kwargs.get('solution', kwargs.get('label', [None] * len(prompts)))
        if not isinstance(solutions, list):
            solutions = [solutions] * len(prompts)
        if len(solutions) < len(prompts):
            solutions = solutions + [None] * (len(prompts) - len(solutions))
        if prompt_ids is None:
            prompt_ids = [i // self.num_generations for i in range(len(prompts))]

        format_map = self._check_format(completions_text)
        correct_map = self._check_correctness(completions_text, solutions, format_map)
        length_map = self._compute_lengths(completion_ids, correct_map)
        bonuses = self._compute_confidence_aware_bonus(
            length_map, correct_map, completions_text, logits
        )
        return self._combine_rewards(format_map, correct_map, bonuses)

    def _extract_completion_text(self, completions):
        completions_text = []
        for comp in completions:
            if comp is None:
                completions_text.append("")
            elif isinstance(comp, str):
                completions_text.append(comp)
            elif isinstance(comp, list) and len(comp) > 0:
                if isinstance(comp[0], dict):
                    if 'content' in comp[0]:
                        completions_text.append(comp[0]['content'])
                    elif 'text' in comp[0]:
                        completions_text.append(comp[0]['text'])
                    else:
                        completions_text.append(str(comp[0]))
                else:
                    completions_text.append(str(comp[0]))
            elif isinstance(comp, dict):
                if 'content' in comp:
                    completions_text.append(comp['content'])
                elif 'text' in comp:
                    completions_text.append(comp['text'])
                else:
                    completions_text.append(str(comp))
            else:
                completions_text.append(str(comp))
        return completions_text

    def _check_format(self, completions):
        format_map = []
        for completion in completions:
            if completion is None or not isinstance(completion, str):
                is_formatted = False
            else:
                is_formatted = bool(re.search(r'\\boxed\{.*?\}', completion))
            format_map.append(is_formatted)
        return format_map

    def _check_correctness(self, completions, solutions, format_map):
        correct_map = []
        for completion, solution, is_formatted in zip(completions, solutions, format_map):
            if not is_formatted or solution is None or completion is None:
                correct_map.append(False)
            else:
                try:
                    extracted_response = parse(completion)
                    extracted_solution = parse(solution)
                    correct_map.append(verify(extracted_response, extracted_solution))
                except:
                    correct_map.append(False)
        return correct_map

    def _compute_lengths(self, completion_ids, correct_map):
        length_map = []
        for is_correct, cid in zip(correct_map, completion_ids):
            if is_correct:
                try:
                    length = self.length_judge(cid)
                    length_map.append(float(length))
                except:
                    length_map.append(float('nan'))
            else:
                length_map.append(float('nan'))
        return length_map

    def _compute_confidence_aware_bonus(self, length_map, correct_map, completions_text, logits):
        bonuses = [0.0] * len(length_map)

        if logits is None:
            for i, (is_correct, comp_text, length) in enumerate(zip(correct_map, completions_text, length_map)):
                if not is_correct or np.isnan(length):
                    continue
                quality = evaluate_reasoning_quality(comp_text)
                confidence = quality
                bonuses[i] = self._compute_length_bonus(length, confidence)
        else:
            try:
                confidence = compute_avg_confidence(logits)
                if self.use_entropy:
                    entropy = compute_entropy(logits)
                else:
                    entropy = None
                for i, (is_correct, length) in enumerate(zip(correct_map, length_map)):
                    if not is_correct or np.isnan(length):
                        continue
                    conf = confidence[i].item()
                    if entropy is not None:
                        ent = entropy[i].item()
                        if ent > self.entropy_threshold:
                            effective_confidence = conf * 0.5
                        else:
                            effective_confidence = conf
                    else:
                        effective_confidence = conf
                    bonuses[i] = self._compute_length_bonus(length, effective_confidence)
            except:
                for i, (is_correct, comp_text, length) in enumerate(zip(correct_map, completions_text, length_map)):
                    if not is_correct or np.isnan(length):
                        continue
                    quality = evaluate_reasoning_quality(comp_text)
                    bonuses[i] = self._compute_length_bonus(length, quality)

        return bonuses

    def _compute_length_bonus(self, length, confidence):
        if confidence < self.confidence_threshold:
            if self.optimal_length_min <= length <= self.optimal_length_max:
                return 0.02
            else:
                return 0.0

        if self.optimal_length_min <= length <= self.optimal_length_max:
            return self.optimal_length_reward
        elif length < self.optimal_length_min:
            if length > 50:
                return self.short_length_reward * min(1.0, length / 100)
            else:
                return 0.02
        else:
            if length < self.optimal_length_max * 1.5:
                return self.long_length_reward * max(0,
                                                     1 - (length - self.optimal_length_max) / self.optimal_length_max)
            else:
                return 0.02

    def _combine_rewards(self, format_map, correct_map, bonuses):
        rewards = []
        for is_formatted, is_correct, bonus in zip(format_map, correct_map, bonuses):
            reward = 0.0
            if not is_formatted:
                reward += self.format_penalty
            if is_correct:
                reward += self.answer_reward
            reward += bonus
            rewards.append(reward)
        return rewards


# ============ 6. QualityConfidenceHybridRewarder (混合推荐) ============

@dataclass
class QualityConfidenceHybridRewarder:
    """
    质量筛选 + 置信度感知 混合奖励器 (推荐)
    """
    num_generations: int
    length_judge: callable
    answer_reward: float = 1.0
    format_penalty: float = -0.3
    quality_threshold: float = 0.6
    min_steps: int = 2
    max_steps: int = 8
    confidence_threshold: float = 0.7
    optimal_length_min: float = 150
    optimal_length_max: float = 350
    optimal_length_reward: float = 0.15
    short_length_reward: float = 0.1
    long_length_reward: float = 0.05

    __name__: str = "QualityConfidenceHybridRewarder"

    def __post_init__(self):
        if self.format_penalty > 0:
            raise ValueError(f"format_penalty must be <= 0. Got: {self.format_penalty}")

    def __call__(self, prompts, completions, completion_ids, prompt_ids=None, solutions=None, logits=None, **kwargs):
        completions_text = self._extract_completion_text(completions)
        if solutions is None:
            solutions = kwargs.get('solution', kwargs.get('label', [None] * len(prompts)))
        if not isinstance(solutions, list):
            solutions = [solutions] * len(prompts)
        if len(solutions) < len(prompts):
            solutions = solutions + [None] * (len(prompts) - len(solutions))
        if prompt_ids is None:
            prompt_ids = [i // self.num_generations for i in range(len(prompts))]

        format_map = self._check_format(completions_text)
        correct_map = self._check_correctness(completions_text, solutions, format_map)
        length_map = self._compute_lengths(completion_ids, correct_map)

        quality_map = []
        for is_correct, comp_text in zip(correct_map, completions_text):
            if is_correct:
                quality = evaluate_reasoning_quality(comp_text, self.min_steps, self.max_steps)
                quality_map.append(quality)
            else:
                quality_map.append(0.0)

        adjustments = [0.0] * len(correct_map)

        if logits is not None:
            try:
                confidence = compute_avg_confidence(logits)
            except:
                confidence = None
        else:
            confidence = None

        for i, (is_correct, length, quality, comp_text) in enumerate(
                zip(correct_map, length_map, quality_map, completions_text)):
            if not is_correct or np.isnan(length):
                continue
            if quality < self.quality_threshold:
                continue
            if confidence is not None:
                conf = confidence[i].item()
            else:
                conf = quality
            if conf < self.confidence_threshold:
                if self.optimal_length_min <= length <= self.optimal_length_max:
                    adjustments[i] = 0.05
                continue
            if self.optimal_length_min <= length <= self.optimal_length_max:
                adjustments[i] = self.optimal_length_reward
            elif length < self.optimal_length_min:
                if length > 50:
                    adjustments[i] = self.short_length_reward * min(1.0, length / 100)
                else:
                    adjustments[i] = 0.02
            else:
                if length < self.optimal_length_max * 1.5:
                    adjustments[i] = self.long_length_reward * max(0, 1 - (
                                length - self.optimal_length_max) / self.optimal_length_max)
                else:
                    adjustments[i] = 0.02

        return self._combine_rewards(format_map, correct_map, adjustments)

    def _extract_completion_text(self, completions):
        completions_text = []
        for comp in completions:
            if comp is None:
                completions_text.append("")
            elif isinstance(comp, str):
                completions_text.append(comp)
            elif isinstance(comp, list) and len(comp) > 0:
                if isinstance(comp[0], dict):
                    if 'content' in comp[0]:
                        completions_text.append(comp[0]['content'])
                    elif 'text' in comp[0]:
                        completions_text.append(comp[0]['text'])
                    else:
                        completions_text.append(str(comp[0]))
                else:
                    completions_text.append(str(comp[0]))
            elif isinstance(comp, dict):
                if 'content' in comp:
                    completions_text.append(comp['content'])
                elif 'text' in comp:
                    completions_text.append(comp['text'])
                else:
                    completions_text.append(str(comp))
            else:
                completions_text.append(str(comp))
        return completions_text

    def _check_format(self, completions):
        format_map = []
        for completion in completions:
            if completion is None or not isinstance(completion, str):
                is_formatted = False
            else:
                is_formatted = bool(re.search(r'\\boxed\{.*?\}', completion))
            format_map.append(is_formatted)
        return format_map

    def _check_correctness(self, completions, solutions, format_map):
        correct_map = []
        for completion, solution, is_formatted in zip(completions, solutions, format_map):
            if not is_formatted or solution is None or completion is None:
                correct_map.append(False)
            else:
                try:
                    extracted_response = parse(completion)
                    extracted_solution = parse(solution)
                    correct_map.append(verify(extracted_response, extracted_solution))
                except:
                    correct_map.append(False)
        return correct_map

    def _compute_lengths(self, completion_ids, correct_map):
        length_map = []
        for is_correct, cid in zip(correct_map, completion_ids):
            if is_correct:
                try:
                    length = self.length_judge(cid)
                    length_map.append(float(length))
                except:
                    length_map.append(float('nan'))
            else:
                length_map.append(float('nan'))
        return length_map

    def _combine_rewards(self, format_map, correct_map, adjustments):
        rewards = []
        for is_formatted, is_correct, adj in zip(format_map, correct_map, adjustments):
            reward = 0.0
            if not is_formatted:
                reward += self.format_penalty
            if is_correct:
                reward += self.answer_reward
            reward += adj
            rewards.append(reward)
        return rewards


# ============ Helper functions for length calculation ============

def count_reasoning_tokens(completion_ids: List[int], special_token_ids: List[int] = None) -> int:
    if special_token_ids is None:
        return len(completion_ids)
    count = 0
    for token_id in completion_ids:
        if token_id in special_token_ids:
            count += 1
    return count


def count_thinking_steps(completion: str, step_separator: str = "\\n\\n") -> int:
    if not completion:
        return 0
    return completion.count(step_separator) + 1


# ============ Evaluation Functions ============


def eval_correctness_check(responses: list[str], solutions: list, dataset_name: str) -> tuple:
    """Correctness checking function for evaluation, supporting multiple dataset types."""
    
    # 调试信息（可保留）
    # print(f"[DEBUG] eval_correctness_check called for dataset: {dataset_name}")
    
    MULTI_CHOICE_DATASETS = ['arcc', 'mmlust', 'gpqa_diamond', 'arc_challenge']
    
    # 如果是多选题，使用选项匹配
    if dataset_name in MULTI_CHOICE_DATASETS:
        return _eval_multi_choice(responses, solutions, dataset_name)
    else:
        # 其他数据集使用原有逻辑
        extracted_responses = [parse(r) for r in responses]
        extracted_solutions = [parse(a) for a in solutions]
        is_correct_list = [1.0 if verify(r, a) else 0.0 
                          for r, a in zip(extracted_responses, extracted_solutions)]
        extracted_responses_str = [_safe_str_conversion(r) for r in extracted_responses]
        extracted_solutions_str = [_safe_str_conversion(a) for a in extracted_solutions]
        return is_correct_list, extracted_responses_str, extracted_solutions_str


def _eval_multi_choice(responses: list[str], solutions: list, dataset_name: str) -> tuple:
    """
    多选题评估：通过选项内容匹配来验证答案
    """
    is_correct_list = []
    extracted_responses_str = []
    extracted_solutions_str = []
    
    # 获取选项映射（从数据集中）
    # 由于 eval_correctness_check 没有直接访问 dataset，我们通过 solutions 传递
    # 如果 solutions 是字符串（如 'C'），我们需要从 response 中提取选项内容
    # 更好的方式：在数据加载时，将选项映射作为额外信息传递
    
    for response, solution in zip(responses, solutions):
        # 1. 从 response 中提取答案（数值或选项内容）
        extracted_answer = _extract_answer_from_response(response)
        
        # 2. 从 solution 中提取答案选项
        extracted_solution = _extract_solution_option(solution)
        
        # 3. 判断是否匹配
        is_correct = _match_answer_with_options(extracted_answer, extracted_solution, response)
        
        is_correct_list.append(1.0 if is_correct else 0.0)
        extracted_responses_str.append(str(extracted_answer))
        extracted_solutions_str.append(str(extracted_solution))
    
    return is_correct_list, extracted_responses_str, extracted_solutions_str


def _extract_answer_from_response(response: str) -> str:
    """
    从模型响应中提取答案
    支持格式：
    - \boxed{C} -> C
    - \boxed{33.4} -> 33.4
    - Answer: C -> C
    - C -> C
    - The answer is C -> C
    """
    import re
    
    if not response:
        return ""
    
    # 1. 尝试提取 \boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        content = boxed_match.group(1).strip()
        # 如果内容是单个字母，直接返回
        if len(content) == 1 and content.upper() in 'ABCD':
            return content.upper()
        return content
    
    # 2. 尝试提取 "answer": "C" 或 answer: C
    answer_match = re.search(r'"?answer"?\s*[:=]\s*"?([A-Da-d])"?' , response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # 3. 尝试提取选项字母 (A, B, C, D) 在句末
    # 查找最后的 A/B/C/D
    last_letter = _get_last_option(response)
    if last_letter:
        return last_letter
    
    # 4. 尝试提取数值（用于匹配选项内容）
    number_match = re.search(r'[-+]?\d*\.?\d+', response)
    if number_match:
        return number_match.group()
    
    return ""


def _extract_solution_option(solution) -> str:
    """
    从 ground truth 中提取选项字母
    """
    if solution is None:
        return ""
    
    if isinstance(solution, str):
        solution = solution.strip()
        # 如果是单个字母
        if len(solution) == 1 and solution.upper() in 'ABCD':
            return solution.upper()
        # 如果包含字母
        for letter in 'ABCD':
            if letter in solution:
                return letter
        return solution
    
    # 如果是列表或元组
    if isinstance(solution, (list, tuple)):
        for item in solution:
            result = _extract_solution_option(item)
            if result:
                return result
    
    return str(solution)


def _match_answer_with_options(extracted_answer: str, solution_letter: str, response: str) -> bool:
    """
    通过选项内容匹配来判断答案是否正确
    """
    import re
    
    # 1. 如果提取的答案就是选项字母，直接比较
    if extracted_answer and len(extracted_answer) == 1 and extracted_answer.upper() in 'ABCD':
        return extracted_answer.upper() == solution_letter.upper()
    
    # 2. 从 response 中提取选项内容，然后匹配对应的选项字母
    # 例如：response 中包含 "~0.32"，选项中有 "~0.32"，对应 A
    
    # 尝试从 response 中提取关键数值/文本
    # 这里需要从数据集的 choices 中获取映射，但 eval_correctness_check 无法直接访问
    # 所以我们通过 response 中的选项字母和内容的对应关系来判断
    
    # 3. 如果提取的是数值，尝试从 response 中找对应的选项字母
    # 使用正则匹配 "A. ~0.32" 或 "A ~0.32" 或 "A) ~0.32"
    if extracted_answer:
        # 查找所有选项格式
        option_patterns = [
            r'([A-D])\s*[\.\)]\s*([^\n,]+)',  # A. xxx 或 A) xxx
            r'([A-D])\s*=\s*([^\n,]+)',        # A = xxx
            r'([A-D])\s+([^\n,]+)',            # A xxx
        ]
        
        for pattern in option_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for letter, content in matches:
                content = content.strip()
                # 如果提取的答案在选项内容中
                if extracted_answer in content or content in extracted_answer:
                    return letter.upper() == solution_letter.upper()
                # 如果内容相似（去除空格、特殊字符）
                if _similar_strings(extracted_answer, content):
                    return letter.upper() == solution_letter.upper()
    
    # 4. 最后尝试：从 response 中找选项字母
    last_letter = _get_last_option(response)
    if last_letter:
        return last_letter == solution_letter.upper()
    
    return False


def _similar_strings(s1: str, s2: str, threshold: float = 0.8) -> bool:
    """
    判断两个字符串是否相似（用于数值比较）
    """
    import re
    
    # 清理字符串
    def clean(s):
        s = re.sub(r'[~\s,()]', '', s)
        s = re.sub(r'\\', '', s)
        return s.strip()
    
    s1_clean = clean(s1)
    s2_clean = clean(s2)
    
    # 直接比较
    if s1_clean == s2_clean:
        return True
    
    # 数值比较（如果都是数字）
    try:
        num1 = float(re.search(r'[-+]?\d*\.?\d+', s1_clean).group()) if re.search(r'[-+]?\d*\.?\d+', s1_clean) else None
        num2 = float(re.search(r'[-+]?\d*\.?\d+', s2_clean).group()) if re.search(r'[-+]?\d*\.?\d+', s2_clean) else None
        if num1 is not None and num2 is not None:
            # 相对误差小于 5%
            if abs(num1 - num2) / max(abs(num1), abs(num2), 1e-10) < 0.05:
                return True
    except:
        pass
    
    return False


def _get_last_option(text: str) -> str:
    """从文本末尾提取选项字母 (A, B, C, D)"""
    if not text:
        return None
    targets = {'A', 'B', 'C', 'D'}
    for char in reversed(text):
        if char in targets:
            return char
    return None


def _safe_str_conversion(obj) -> str:
    """安全地将对象转换为字符串"""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return ""
        if len(obj) > 1 and isinstance(obj[1], str):
            return obj[1]
        first = obj[0]
        if isinstance(first, str):
            return first
        if isinstance(first, (list, tuple)):
            return _safe_str_conversion(first)
        try:
            return str(first)
        except:
            return ""
    try:
        return str(obj)
    except:
        return ""


# ============ Integration with existing training pipeline ============

def create_enhanced_reward_function(
        num_generations: int = 8,
        accuracy_requirement: float = 0.5,
        length_judge: callable = None,
        **kwargs
):
    if length_judge is None:
        length_judge = lambda ids: len(ids)

    rewarder = RelativeLengthRewarder(
        num_generations=num_generations,
        length_judge=length_judge,
        accuracy_requirement=accuracy_requirement,
        **kwargs
    )
    return rewarder