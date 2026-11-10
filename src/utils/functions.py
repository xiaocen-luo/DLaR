import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from typing import Optional
import random
import numpy as np


def CE_loss_from_model(
    model: Qwen2ForCausalLM,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    labels_topk_ids: torch.Tensor,
    labels_topk_probs: torch.Tensor,
    labels_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the cross-entropy loss between model predictions and a given top-k distribution at specified label positions.
    **Note**: This function implements the standard Causal LM loss computation, where logits and labels are shifted by one position.
    That is, the logits at position i predict the token at position i+1 in labels.

    Args:
        model (Qwen2ForCausalLM): Hugging Face Qwen2ForCausalLM model.
        inputs_embeds (torch.Tensor): Input embeddings, shape (batch_size, prompt_len + completion_len, hidden_size).
        attention_mask (torch.Tensor): Attention mask, shape (batch_size, prompt_len + completion_len).
        labels_topk_ids (torch.Tensor): Top-k token IDs for target distribution, shape (batch_size, prompt_len + completion_len, topk).
        labels_topk_probs (torch.Tensor): Probabilities for the top-k token IDs, shape (batch_size, prompt_len + completion_len, topk).
        labels_mask (torch.Tensor): Label mask, shape (batch_size, prompt_len + completion_len).

    Returns:
        torch.Tensor: masked_loss and outputs
    """
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=False,
        output_attentions=False
    )
    logits = outputs.logits

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels_topk_ids = labels_topk_ids[:, 1:, :].contiguous()
    shifted_labels_topk_probs = labels_topk_probs[:, 1:, :].contiguous()
    shifted_labels_mask = labels_mask[:, 1:].contiguous()

    num_labels = shifted_labels_mask.sum()
    if num_labels == 0:
        return torch.tensor(0.0, device=inputs_embeds.device)

    log_probs = F.log_softmax(shifted_logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs, 2, shifted_labels_topk_ids)
    loss_per_position = -torch.sum(shifted_labels_topk_probs * gathered_log_probs, dim=-1)
    masked_loss = loss_per_position * shifted_labels_mask.to(loss_per_position.dtype)

    return masked_loss, outputs


def shuffle_tensor_dict(tensor_dict: dict[str, Optional[torch.Tensor]]) -> dict[str, Optional[torch.Tensor]]:
    """Shuffles a dictionary of tensors along the first dimension in unison."""
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    batch_size = first_tensor.shape[0]
    permutation = torch.randperm(batch_size)
    return {key: tensor[permutation] if tensor is not None else None for key, tensor in tensor_dict.items()}


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts."""
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


def prepare_pretraining_inputs(
    inputs: dict,
    pad_token_id: int = 0,
    ignore_index: int = -100
) -> dict:
    """Convert input data containing prompt and completion into model training format."""
    prompt_ids = inputs['prompt_ids']
    prompt_mask = inputs['prompt_mask']
    completion_ids = inputs['completion_ids']
    completion_mask = inputs['completion_mask']
    advantages = inputs['advantages']

    prompt_lengths = prompt_mask.sum(dim=1)
    completion_lengths = completion_mask.sum(dim=1)

    batch_size = prompt_ids.size(0)

    unpadded_input_ids_list = []
    unpadded_labels_list = []

    for i in range(batch_size):
        prompt_len = int(prompt_lengths[i].item())
        valid_prompt = prompt_ids[i, -prompt_len:]

        completion_len = int(completion_lengths[i].item())
        valid_completion = completion_ids[i, :completion_len]

        combined_ids = torch.cat((valid_prompt, valid_completion), dim=0)
        unpadded_input_ids_list.append(combined_ids)

        prompt_labels = torch.full_like(valid_prompt, ignore_index)
        combined_labels = torch.cat((prompt_labels, valid_completion), dim=0)
        unpadded_labels_list.append(combined_labels)

    max_length = max(len(seq) for seq in unpadded_input_ids_list)

    device = prompt_ids.device
    dtype = prompt_ids.dtype

    padded_input_ids = torch.full((batch_size, max_length),
                                  fill_value=pad_token_id,
                                  dtype=dtype,
                                  device=device)

    padded_labels = torch.full((batch_size, max_length),
                               fill_value=ignore_index,
                               dtype=dtype,
                               device=device)

    attention_mask = torch.zeros((batch_size, max_length),
                                 dtype=torch.long,
                                 device=device)

    for i in range(batch_size):
        seq_len = len(unpadded_input_ids_list[i])
        padded_input_ids[i, :seq_len] = unpadded_input_ids_list[i]
        padded_labels[i, :seq_len] = unpadded_labels_list[i]
        attention_mask[i, :seq_len] = 1

    return {
        'input_ids': padded_input_ids,
        'attention_mask': attention_mask,
        'labels': padded_labels,
        'advantages': advantages,
    }


def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the standard deviation of a tensor, ignoring NaNs."""
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)
    count = torch.sum(~torch.isnan(tensor))
    variance *= count / (count - 1)
    return torch.sqrt(variance)


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the minimum value of a tensor, ignoring NaNs."""
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the maximum value of a tensor, ignoring NaNs."""
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


def truncate_with_protected_tokens(
    ids: torch.Tensor, mask: torch.Tensor, target_length: int, protected_tokens: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate tensors to target length while preserving protected tokens."""
    protected_set = set(protected_tokens)

    def process_sequence(ids, mask):
        is_protected = torch.tensor([x.item() in protected_set for x in ids])
        is_non_protected = ~is_protected

        num_protected = is_protected.sum().item()
        num_non_protected_needed = target_length - num_protected

        if num_non_protected_needed < 0:
            raise ValueError(
                f"target_length ({target_length}) is too small for the protected tokens ({num_protected} tokens)."
            )

        non_protected_indices = torch.where(is_non_protected)[0]
        keep_non_protected = torch.zeros_like(is_non_protected)
        if num_non_protected_needed > 0:
            keep_indices = non_protected_indices[-num_non_protected_needed:]
            keep_non_protected[keep_indices] = True

        keep_mask = is_protected | keep_non_protected
        return ids[keep_mask], mask[keep_mask]

    truncated_seq = []
    truncated_mask = []

    for i in range(ids.shape[0]):
        new_ids, new_mask = process_sequence(ids[i], mask[i])
        truncated_seq.append(new_ids)
        truncated_mask.append(new_mask)

    return torch.stack(truncated_seq), torch.stack(truncated_mask)


def entropy_from_logits(logits, chunk_size: int = 1) -> torch.Tensor:
    """Compute the Shannon entropy (in nats) for each row of logits."""
    per_token_entropies = []
    for logits_chunk in logits.split(chunk_size, dim=0):
        logps = F.log_softmax(logits_chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        per_token_entropies.extend(chunk_entropy)

    per_token_entropies = torch.stack(per_token_entropies)
    return per_token_entropies


def selective_log_softmax(logits, index) -> torch.Tensor:
    """A memory-efficient implementation of the common `log_softmax -> gather` operation."""
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values
    else:
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def set_deterministic_seed(seed: int):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print(f"Deterministic seed {seed} has been set.")
