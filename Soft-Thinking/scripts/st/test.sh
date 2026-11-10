python run_sglang_softthinking.py \
    --dataset "amc23" \
    --model_name "/public/home/xiangyuduan/xcluo/lepo-main/outputs/sweep/linear_a09/llama-CLaR-0shot-32-gumbel015-dapo-en-train-20261010_2110/checkpoint-200/" \
    --max_topk 5 \
    --max_generated_tokens 2048 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.0 \
    --after_thinking_temperature 0.6 \
    --after_thinking_top_p 0.95 \
    --after_thinking_top_k 30 \
    --after_thinking_min_p 0.0 \
    --mem_fraction_static 0.3 \
    --start_idx 0 \
    --end_idx 10000 \
    --num_gpus 1 \
    --num_samples 8 \
#    --enable_soft_thinking \
#    --add_noise_gumbel_softmax \
#    --gumbel_softmax_temperature 0.5
#    --add_noise_dirichlet \
#    --dirichlet_alpha 4.0
#    --model_name "/public/home/xiangyuduan/models/hf/Llama-3.2-1B-Instruct/" \
# /public/home/xiangyuduan/models/hf/DeepSeek-R1-Distill-Qwen-1.5B/
#gsm8k    math500   minervamath   amc23    gpqa_diamond
