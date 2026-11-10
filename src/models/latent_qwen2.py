import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from typing import Optional, List, Union, Callable, Dict
from transformers.utils import logging
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.streamers import BaseStreamer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
import torch.distributed as dist
import inspect
import os
import warnings
import numpy as np

from torch.distributions import Dirichlet
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
)

GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]

logger = logging.get_logger(__name__)


class LatentGenerationConfig(GenerationConfig):
    """Extended generation config with latent generation parameters."""

    def __init__(self, **kwargs):
        # 提取 latent 参数
        self.latent_length = kwargs.pop("latent_length", 20)
        self.use_latent = kwargs.pop("use_latent", True)
        self.do_latent_sample = kwargs.pop("do_latent_sample", False)
        self.do_discrete_sample = kwargs.pop("do_discrete_sample", True)
        self.noise_type = kwargs.pop("noise_type", "gumbel")
        self.noise_strength = kwargs.pop("noise_strength", 0.01)

        # 置信度引导参数
        self.use_confidence_guided_noise = kwargs.pop("use_confidence_guided_noise", False)
        self.confidence_weight = kwargs.pop("confidence_weight", 0.7)
        self.entropy_weight = kwargs.pop("entropy_weight", 0.3)
        self.min_noise_strength = kwargs.pop("min_noise_strength", 0.01)
        self.max_noise_strength = kwargs.pop("max_noise_strength", 0.3)
        self.noise_schedule = kwargs.pop("noise_schedule", None)
        self.noise_warmup_steps = kwargs.pop("noise_warmup_steps", 1000)
        self.noise_start_strength = kwargs.pop("noise_start_strength", None)
        self.noise_end_strength = kwargs.pop("noise_end_strength", None)
        self.use_adaptive_noise = kwargs.pop("use_adaptive_noise", True)

        super().__init__(**kwargs)


class LatentQwen2ForCausalLM(Qwen2ForCausalLM):
    """Qwen2 model with latent generation support and confidence-guided stochasticity."""

    def __init__(self, config):
        super().__init__(config)
        # 用于跟踪训练步数（在训练循环中更新）
        self.current_step = 0

    def set_current_step(self, step: int):
        """设置当前训练步数，用于动态噪声调度"""
        self.current_step = step

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """Generate token sequences with latent generation mode support."""

        if custom_generate is not None:
            trust_remote_code = kwargs.pop("trust_remote_code", None)
            global_keys_to_exclude = {"self", "kwargs"}
            generate_arguments = {key: value for key, value in locals().items() if key not in global_keys_to_exclude}
            generate_arguments.update(kwargs)
            custom_generate_function = self.load_custom_generate(
                custom_generate, trust_remote_code=trust_remote_code, **kwargs
            )
            return custom_generate_function(model=self, **generate_arguments)

        tokenizer = kwargs.pop("tokenizer", None)
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)

        # 从kwargs中获取当前步数
        self.current_step = kwargs.pop("current_step", self.current_step)

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        if not self.config.is_encoder_decoder:
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
        )

        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        model_kwargs["use_cache"] = generation_config.use_cache

        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")
            if generation_config.cache_implementation in ["static", "hybrid", "sliding_window"]:
                raise ValueError("assisted generate is not supported with Static cache classes`")
            if self._is_stateful:
                raise ValueError(
                    f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
                )

            candidate_generator = self._get_candidate_generator(
                generation_config=generation_config,
                input_ids=input_ids,
                inputs_tensor=inputs_tensor,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                target_tokenizer=tokenizer,
                assistant_tokenizer=assistant_tokenizer,
                model_kwargs=model_kwargs,
            )

            result = self._assisted_decoding(
                input_ids,
                candidate_generator=candidate_generator,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        if (
            generation_config.return_legacy_cache is True
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result

    def _compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算预测置信度

        Args:
            logits: [batch_size, vocab_size] 原始logits

        Returns:
            confidence: [batch_size] 每个样本的置信度分数 (0-1)
        """
        probs = F.softmax(logits, dim=-1)
        # 使用最大概率作为置信度
        max_probs, _ = probs.max(dim=-1)
        return max_probs

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算预测熵（不确定性度量）

        Args:
            logits: [batch_size, vocab_size] 原始logits

        Returns:
            entropy: [batch_size] 每个样本的熵值
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return entropy

    def _apply_noise_decay_only(
        self,
        generation_config: "LatentGenerationConfig",
        step: int,
        base_noise: float
    ) -> float:
        """
        仅应用全局噪声衰减，不包含置信度引导

        Args:
            generation_config: 生成配置
            step: 当前训练步数
            base_noise: 基础噪声强度

        Returns:
            decayed_noise: 衰减后的噪声强度
        """
        # 如果未启用衰减，直接返回基础噪声
        if getattr(generation_config, 'noise_schedule', None) is None:
            return base_noise

        warmup_steps = getattr(generation_config, 'noise_warmup_steps', 1000)
        start_strength = getattr(generation_config, 'noise_start_strength', base_noise)
        end_strength = getattr(generation_config, 'noise_end_strength', base_noise * 0.1)
        progress = min(step / warmup_steps, 1.0)

        if generation_config.noise_schedule == 'linear':
            decayed = start_strength - (start_strength - end_strength) * progress
        elif generation_config.noise_schedule == 'cosine':
            decayed = end_strength + 0.5 * (start_strength - end_strength) * (1 + np.cos(np.pi * progress))
        elif generation_config.noise_schedule == 'step':
            if progress < 0.3:
                decayed = start_strength
            elif progress < 0.6:
                decayed = start_strength * 0.5
            else:
                decayed = end_strength
        else:
            decayed = base_noise

        return decayed

    def _get_adaptive_noise_strength(
        self,
        logits: torch.Tensor,
        generation_config: "LatentGenerationConfig",
        step: int,
        noise_type: str = "gumbel"
    ) -> torch.Tensor:
        """
        计算自适应的噪声强度

        核心思想：
        1. 高置信度位置 -> 低噪声（保持已有知识）
        2. 低置信度位置 -> 高噪声（鼓励探索）
        3. 结合训练步数进行退火

        Args:
            logits: [batch_size, vocab_size] 原始logits
            generation_config: 生成配置
            step: 当前训练步数
            noise_type: 噪声类型

        Returns:
            adaptive_noise: [batch_size, 1] 每个样本的噪声强度
        """
        # 1. 计算置信度
        confidence = self._compute_confidence(logits)  # [batch_size]
        entropy = self._compute_entropy(logits)  # [batch_size]

        # 2. 归一化熵 (假设熵的范围在0 ~ log(vocab_size)之间)
        vocab_size = logits.size(-1)
        normalized_entropy = entropy / torch.log(torch.tensor(vocab_size, dtype=torch.float32))

        # 3. 基于置信度的基础噪声调制
        # 置信度越高，噪声越小；置信度越低，噪声越大
        # confidence=1.0 -> noise_scale=0.0; confidence=0.0 -> noise_scale=1.0
        confidence_scale = 1.0 - confidence

        # 4. 基于熵的额外调制
        # 熵越高（更不确定），噪声越大
        entropy_scale = normalized_entropy

        # 5. 组合两种调制 (加权平均)
        # 可以调整权重来控制置信度和熵的相对重要性
        confidence_weight = getattr(generation_config, 'confidence_weight', 0.7)
        entropy_weight = getattr(generation_config, 'entropy_weight', 0.3)
        combined_scale = confidence_weight * confidence_scale + entropy_weight * entropy_scale

        # 6. 基础噪声强度（可配置）
        base_noise = generation_config.noise_strength

        # 7. 训练步数退火
        decayed_base = self._apply_noise_decay_only(generation_config, step, base_noise)

        # 8. 计算最终自适应噪声强度
        # 确保噪声强度在合理范围内
        min_noise = getattr(generation_config, 'min_noise_strength', 0.01)
        max_noise = getattr(generation_config, 'max_noise_strength', decayed_base * 2.0)

        # 自适应噪声 = 退火基础噪声 * (1 + 调制因子)
        # 调制因子范围: 0 ~ 1, 使得噪声在 [base_noise, 2*base_noise] 范围内变化
        modulation = combined_scale.unsqueeze(-1)  # [batch_size, 1]
        adaptive_noise = decayed_base * (1 + modulation * 0.5)  # 0.5倍调节范围

        # 裁剪到安全范围
        adaptive_noise = torch.clamp(adaptive_noise, min=min_noise, max=max_noise)

        return adaptive_noise

    def _apply_confidence_guided_noise(
        self,
        next_token_scores: torch.Tensor,
        generation_config: "LatentGenerationConfig",
        step: int,
        noise_type: str = "gumbel",
        use_adaptive: bool = True
    ) -> torch.Tensor:
        """
        应用置信度引导的噪声到token scores

        Args:
            next_token_scores: [batch_size, vocab_size] 原始token scores
            generation_config: 生成配置
            step: 当前训练步数
            noise_type: 噪声类型 ('gumbel', 'gaussian', 'dirichlet', 'soft')
            use_adaptive: True=使用自适应(每步变化), False=仅使用衰减(固定噪声)

        Returns:
            probs: [batch_size, vocab_size] 添加噪声后的概率分布
        """
        if use_adaptive:
            # 原始路径：自适应噪声强度（包含置信度 + 衰减）
            adaptive_noise_strength = self._get_adaptive_noise_strength(
                next_token_scores, generation_config, step, noise_type
            )
        else:
            # 新路径：仅衰减，无自适应
            base_noise = generation_config.noise_strength
            decayed_noise = self._apply_noise_decay_only(generation_config, step, base_noise)
            # 对所有样本使用相同的衰减后噪声
            adaptive_noise_strength = torch.full(
                (next_token_scores.size(0), 1),
                decayed_noise,
                device=next_token_scores.device,
                dtype=torch.float32
            )

        # 根据噪声类型应用不同的噪声
        if noise_type == "gaussian":
            # 高斯噪声
            clean_scores = next_token_scores.masked_fill(next_token_scores == float('-inf'), 0.0)
            count = (next_token_scores != float('-inf')).sum(dim=-1, keepdim=True)
            rmsnorm = torch.sqrt((clean_scores ** 2).sum(dim=-1, keepdim=True) / count)
            sigma = adaptive_noise_strength * rmsnorm
            noise = sigma * torch.randn_like(next_token_scores)
            next_token_scores_noisy = next_token_scores + noise
            probs = F.softmax(next_token_scores_noisy, dim=-1, dtype=self.dtype)

        elif noise_type == "gumbel":
            # Gumbel-Softmax噪声 (使用自适应温度)
            # 温度越高，分布越平滑（更随机）
            # 温度越低，分布越sharp（更确定）
            temperature = adaptive_noise_strength  # [batch_size, 1]
            # 对每个样本使用不同的温度
            probs = []
            for i in range(next_token_scores.size(0)):
                sample_probs = F.gumbel_softmax(
                    next_token_scores[i:i + 1],
                    tau=temperature[i].item(),
                    hard=False
                )
                probs.append(sample_probs)
            probs = torch.cat(probs, dim=0).to(self.dtype)

        elif noise_type == "dirichlet":
            # Dirichlet噪声
            total_concentration = adaptive_noise_strength
            original_probs = F.softmax(next_token_scores, dim=-1)
            alpha_params = original_probs * total_concentration + 1e-6
            probs_list = []
            for i in range(next_token_scores.size(0)):
                dist_obj = Dirichlet(alpha_params[i])
                sample_probs = dist_obj.sample().unsqueeze(0)
                probs_list.append(sample_probs)
            probs = torch.cat(probs_list, dim=0).to(self.dtype)

        elif noise_type == "soft":
            # 软噪声 (在embedding空间加噪声)
            probs = F.softmax(next_token_scores, dim=-1, dtype=self.dtype)
            # 对每个样本使用不同的噪声强度
            for i in range(next_token_scores.size(0)):
                # 在概率空间加噪声
                noise = adaptive_noise_strength[i] * torch.randn_like(probs[i])
                probs[i] = F.softmax(torch.log(probs[i] + 1e-10) + noise, dim=-1)

        else:
            raise ValueError(f"Unsupported noise_type: {noise_type}. Please provide a valid noise_type.")

        return probs

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: "LogitsProcessorList",
        stopping_criteria: "StoppingCriteriaList",
        generation_config: "LatentGenerationConfig",
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Core sampling method with confidence-guided latent generation."""

        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

        do_latent_sample = generation_config.do_latent_sample
        do_discrete_sample = generation_config.do_discrete_sample
        top_k = generation_config.top_k

        # ===== 读取自适应控制开关 =====
        use_adaptive_noise = getattr(generation_config, 'use_adaptive_noise', True)

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        embedding_layer = self.get_input_embeddings()
        accumulated_embeds = embedding_layer(input_ids)
        next_token_embeds = accumulated_embeds

        vocab_size = embedding_layer.weight.shape[0]

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        accumulated_topk_probs = torch.zeros(
            (batch_size, seq_len, top_k),
            dtype=accumulated_embeds.dtype,
            device=device
        )
        accumulated_topk_probs[:, :, 0] = 1.0

        accumulated_topk_ids = torch.zeros(
            (batch_size, seq_len, top_k),
            dtype=input_ids.dtype,
            device=device
        )
        accumulated_topk_ids[:, :, 0] = input_ids

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        num_completion_tokens = 0

        # 获取当前步数
        current_step = self.current_step

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            current_input_ids = model_inputs.pop("input_ids")
            model_inputs["inputs_embeds"] = next_token_embeds

            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            use_latent = generation_config.use_latent
            latent_length = generation_config.latent_length

            # 检查是否使用置信度引导的随机性
            use_confidence_guided = getattr(generation_config, 'use_confidence_guided_noise', False)

            if use_latent and num_completion_tokens < latent_length:
                # 获取噪声类型
                noise_type = generation_config.noise_type

                if use_confidence_guided:
                    # ===== 使用置信度引导的随机性 =====
                    probs = self._apply_confidence_guided_noise(
                        next_token_scores,
                        generation_config,
                        current_step,
                        noise_type,
                        use_adaptive=use_adaptive_noise
                    )
                else:
                    # ===== 原始随机性方法（LEPO） =====
                    if noise_type == "gaussian":
                        clean_scores = next_token_scores.masked_fill(next_token_scores == float('-inf'), 0.0)
                        count = (next_token_scores != float('-inf')).sum(dim=-1, keepdim=True)
                        rmsnorm = torch.sqrt((clean_scores ** 2).sum(dim=-1, keepdim=True) / count)
                        sigma = generation_config.noise_strength * rmsnorm
                        noise = sigma * torch.randn_like(next_token_scores)
                        next_token_scores = next_token_scores + noise
                        probs = F.softmax(next_token_scores, dim=-1, dtype=self.dtype)
                    elif noise_type == "dirichlet":
                        total_concentration = generation_config.noise_strength
                        original_probs = F.softmax(next_token_scores, dim=-1)
                        alpha_params = original_probs * total_concentration + 1e-6
                        dist_obj = Dirichlet(alpha_params)
                        probs = dist_obj.sample().to(self.dtype)
                    elif noise_type == "gumbel":
                        probs = F.gumbel_softmax(
                            next_token_scores,
                            tau=generation_config.noise_strength,
                            hard=False
                        ).to(self.dtype)
                    elif noise_type == 'soft':
                        probs = F.softmax(next_token_scores, dim=-1, dtype=self.dtype)
                    else:
                        raise ValueError(f"Unsupported noise_type: {noise_type}")

                # 计算embedding
                embedding_matrix = self.get_input_embeddings().weight
                next_token_embeds = torch.matmul(probs.unsqueeze(1), embedding_matrix)
                next_token_probs = probs.unsqueeze(1)
                next_token_topk_probs, next_token_topk_ids = torch.topk(next_token_probs, k=top_k, dim=-1)

                # 对于'soft'噪声类型，在embedding空间添加额外噪声
                if noise_type == 'soft' and not use_confidence_guided:
                    def rmsnorm(x):
                        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
                        return rms

                    sigma = generation_config.noise_strength * rmsnorm(next_token_embeds)
                    noise = sigma * torch.randn_like(next_token_embeds)
                    next_token_embeds = next_token_embeds + noise

                # 采样或取最大
                if do_latent_sample:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

            else:
                # 离散token生成
                if do_discrete_sample:
                    probs = F.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                next_token_embeds = embedding_layer(next_tokens.unsqueeze(1))
                next_token_probs = F.one_hot(next_tokens.unsqueeze(1), num_classes=vocab_size).to(
                    dtype=accumulated_embeds.dtype)
                next_token_topk_probs, next_token_topk_ids = torch.topk(next_token_probs, k=top_k, dim=-1)

            num_completion_tokens += 1

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            accumulated_embeds = torch.cat([accumulated_embeds, next_token_embeds], dim=1)
            accumulated_topk_probs = torch.cat([accumulated_topk_probs, next_token_topk_probs], dim=1)
            accumulated_topk_ids = torch.cat([accumulated_topk_ids, next_token_topk_ids], dim=1)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            del outputs, next_token_probs

        if streamer is not None:
            streamer.end()

        return {
            "ids": input_ids,
            "embeds": accumulated_embeds,
            "topk_probs": accumulated_topk_probs,
            "topk_ids": accumulated_topk_ids
        }