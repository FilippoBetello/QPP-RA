python rank_llm/src/rank_llm/scripts/run_rank_llm.py  \
    --model_path='castorini/rank_vicuna_7b_v1' \
    --top_k_candidates=20 \
    --dataset=covid \
    --retrieval_method=SPLADE++_EnsembleDistil_ONNX \
    --prompt_mode=rank_GPT \
     --context_size=4096 \
     --variable_passages \