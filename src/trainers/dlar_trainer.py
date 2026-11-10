# src/trainers/dlar_trainer.py

from trl import GRPOConfig, GRPOTrainer
from typing import Any, Callable, Optional, Union
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from accelerate.utils import gather_object
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.trainer.utils import pad
from trl.models import unwrap_model_for_generation
from contextlib import nullcontext
from datasets import Dataset, IterableDataset
from peft import PeftConfig
import copy
import re
import numpy as np

from ..models import LatentGenerationConfig
from ..utils import (
    CE_loss_from_model,
    nanstd,
    nanmin,
    nanmax,
    truncate_with_protected_tokens,
    entropy_from_logits,
    selective_log_softmax,
)

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class DLaRConfig(GRPOConfig):
    """LEPO training configuration."""
    pass


class DLaRTrainer(GRPOTrainer):
    """LEPO trainer with confidence tracking."""

    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None,
                                                                                                               None),
            peft_config: Optional["PeftConfig"] = None,
    ):
        super().__init__(
            model,
            reward_funcs,
            args,
            train_dataset,
            eval_dataset,
            processing_class,
            reward_processing_classes,
            callbacks,
            optimizers,
            peft_config,
        )
        generation_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "pad_token_id": processing_class.pad_token_id,
            "bos_token_id": processing_class.bos_token_id,
            "eos_token_id": processing_class.eos_token_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "cache_implementation": args.cache_implementation,
        }
        if args.generation_kwargs is not None:
            generation_kwargs.update(args.generation_kwargs)
        self.generation_config = LatentGenerationConfig(**generation_kwargs)

    def _generate_and_score_completions(
            self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        original_prompts = copy.deepcopy(prompts)

        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}
            for prompt in prompts:
                if isinstance(prompt, list):
                    for message in prompt:
                        if not isinstance(message, dict):
                            continue
                        content = message.get("content")
                        role = message.get("role")
                        if isinstance(content, str):
                            if role == "user":
                                message["content"] = [{"type": "image"}, {"type": "text", "text": content}]
                            elif role == "system":
                                message["content"] = [{"type": "text", "text": content}]

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
            protected = [token for token in protected if token is not None]
            prompt_ids, prompt_mask = truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )

            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]

            if self.image_token is not None:
                prompts_text = [
                    re.sub(rf"({re.escape(self.image_token)})+", self.image_token, text) for text in prompts_text
                ]

        # Regular generation path
        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask

            prompt_completion_outputs = unwrapped_model.generate(
                prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config, disable_compile=True
            )
            if isinstance(prompt_completion_outputs, dict):
                prompt_completion_ids = prompt_completion_outputs['ids']
                prompt_completion_embeds = prompt_completion_outputs['embeds']
                prompt_completion_topk_probs = prompt_completion_outputs['topk_probs']
                prompt_completion_topk_ids = prompt_completion_outputs['topk_ids']
                # ===== 新增：读取噪声强度历史 =====
                prompt_completion_noise_history = prompt_completion_outputs.get('noise_strength_history', None)
            else:
                prompt_completion_ids = prompt_completion_outputs
                prompt_completion_embeds = None
                prompt_completion_topk_probs = None
                prompt_completion_topk_ids = None
                # ===== 新增 =====
                prompt_completion_noise_history = None

        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        completion_lengths = completion_mask.sum(1)

        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                        image_sizes=prompt_inputs.get("image_sizes"),
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            pixel_values=prompt_inputs.get("pixel_values"),
                            image_grid_thw=prompt_inputs.get("image_grid_thw"),
                            pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                            image_sizes=prompt_inputs.get("image_sizes"),
                        )
            else:
                ref_per_token_logps = None

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list)

        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # ===== 收集 confidence 并直接记录到 _metrics（不添加到 output） =====
        if prompt_completion_topk_probs is not None:
            # prompt_completion_topk_probs: [batch_size, seq_len, top_k]
            topk_probs = prompt_completion_topk_probs
            # 取 top-1 概率
            top1_probs = topk_probs[:, :, 0]  # [batch_size, seq_len]
            prompt_len = prompt_length
            completion_probs = top1_probs[:, prompt_len:]  # [batch_size, completion_len]
            # 过滤 padding（使用 completion_mask）
            completion_mask_expanded = completion_mask
            valid_probs = completion_probs * completion_mask_expanded
            valid_counts = completion_mask_expanded.sum(dim=1).clamp(min=1)
            per_seq_confidence = valid_probs.sum(dim=1) / valid_counts  # [batch_size]
            avg_confidence = per_seq_confidence.mean().item()

            # 计算 confidence 的标准差
            per_seq_std = ((valid_probs - per_seq_confidence.unsqueeze(1)) ** 2 * completion_mask_expanded).sum(
                dim=1) / valid_counts
            avg_confidence_std = per_seq_std.sqrt().mean().item()

            # 聚合所有 rank
            agg_confidence = self.accelerator.gather(torch.tensor([avg_confidence], device=device))
            mean_confidence = agg_confidence.mean().item()
            self._metrics[mode]["confidence/mean"].append(mean_confidence)

            agg_confidence_std = self.accelerator.gather(torch.tensor([avg_confidence_std], device=device))
            mean_confidence_std = agg_confidence_std.mean().item()
            self._metrics[mode]["confidence/std"].append(mean_confidence_std)

        # ===== 新增：记录 adaptive noise strength =====
        if prompt_completion_noise_history is not None and len(prompt_completion_noise_history) > 0:
            mean_noise = float(np.mean(prompt_completion_noise_history))
            agg_noise = self.accelerator.gather(torch.tensor([mean_noise], device=device))
            mean_noise_all = agg_noise.mean().item()
            self._metrics[mode]["adaptive_noise/mean"].append(mean_noise_all)

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if has_images:
            self._logs["image"].extend(gather_object(images))

        # ===== 注意：不在这里添加 confidence！保持 output 的所有值都是可索引的张量/列表 =====
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "prompt_completion_embeds": prompt_completion_embeds,
            "prompt_completion_topk_probs": prompt_completion_topk_probs,
            "prompt_completion_topk_ids": prompt_completion_topk_ids,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in prompt_inputs:
            output["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        if "pixel_attention_mask" in prompt_inputs:
            output["pixel_attention_mask"] = prompt_inputs["pixel_attention_mask"]
        if "image_sizes" in prompt_inputs:
            output["image_sizes"] = prompt_inputs["image_sizes"]
        return output

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        labels = input_ids.clone()
        prompt_len = prompt_ids.size(1)
        completion_len = completion_ids.size(1)

        advantages = inputs['advantages']
        labels[:, :prompt_len] = -100
        labels[attention_mask == 0] = -100

        inputs_embeds = inputs["prompt_completion_embeds"]
        labels_topk_ids, labels_topk_probs = inputs["prompt_completion_topk_ids"], inputs[
            "prompt_completion_topk_probs"]
        labels_mask = attention_mask.clone()
        labels_mask[:, :prompt_len] = 0

        ce_per_token_loss, outputs = CE_loss_from_model(
            model=model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels_topk_ids=labels_topk_ids,
            labels_topk_probs=labels_topk_probs,
            labels_mask=labels_mask,
        )
        ce_per_token_loss = ce_per_token_loss[:, -logits_to_keep:]
        per_token_logps, entropies = self._get_per_token_logps_and_entropies_from_logits(
            input_ids,
            outputs.logits,
            logits_to_keep,
            compute_entropy=True,
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        grpo_per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        grpo_per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        grpo_per_token_loss = -torch.min(grpo_per_token_loss1, grpo_per_token_loss2)

        ce_per_token_loss_mul_advantages = ce_per_token_loss * advantages.unsqueeze(1)

        latent_length = self.args.generation_kwargs['latent_length']

        loss1_prefix = ce_per_token_loss_mul_advantages[:, :latent_length]
        loss2_suffix = grpo_per_token_loss[:, latent_length:]

        per_token_loss = torch.cat([loss1_prefix, loss2_suffix], dim=1)

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    @profiling_decorator
    def _get_per_token_logps_and_entropies_from_logits(
            self,
            input_ids,
            logits,
            logits_to_keep,
            batch_size=None,
            compute_entropy=False,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []

        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        logits = logits / self.temperature

        completion_ids = input_ids[:, -logits_to_keep:]
        logps = selective_log_softmax(logits, completion_ids)
        all_logps.append(logps)

        if compute_entropy:
            with torch.no_grad():
                entropies = entropy_from_logits(logits)
            all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies