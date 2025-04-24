# QPP-RA: Aggregating Large Language Model Rankings
## Abstract
Recent advances in Large Language Models (LLMs) have significantly improved search and recommendation systems. In the field of information retrieval, LLMs are increasingly used as re-rankers to refine the relevance of documents initially retrieved by search algorithms. It is also a common strategy to apply ranking aggregation techniques to improve the performance of the search algorithm.
Existing aggregation methods, such as Borda Count and Reciprocal Rank Fusion (RRF), rely exclusively on the positional information of the retrieved documents and do not take into account the different performance levels of the different models, potentially degrading the overall effectiveness.

To overcome this limitation, we propose a new rank aggregation algorithm called Query Performance Prediction Rank Aggregation (QPP-RA). Our method uses historical model performance data in three different datasets to improve the aggregation process. Our results show that QPP-RA significantly improves the quality and efficiency of search results, significantly outperforming both the single best model and all other aggregation models.

## How to Use This Code

This code is based on the [RankLLM](https://github.com/castorini/rank_llm) library.

**Step 0: Create the conda environment**
Run the command `conda env create -f contracts.yaml`


**Step 1: Insert HuggingFace Token**
Insert your HuggingFace token into the `rank_llm/src/rank_llm/scripts/run_rank_llm.py` function.

### Run Aggregation
To run the aggregation, simply execute the `main.py` script. You can specify the dataset you wish to use. The available datasets are:

- **Trec DL22**
- **Trec News**
- **Trec Covid**

To run the DL22 dataset, modify the following lines in `main.py`:

1. Change the path in lines 5-8 from `results_covid` to `results_dl-22`.
2. Update line 76 similarly.
3. In `utils.py`, change line 14 from `covid` to `dl22`.

Results for NDCG@20 will be saved in the `results_dl-22` directory. To evaluate other metrics at different _k_ values, edit lines 117-130 in `rank_llm/src/rank_llm/retrieve_and_rerank.py`. Note that only one evaluation at a time is permitted.

### Run Base Models

We use three models provided by Castorini and fine-tune an additional model based on Mistral v0.3, called **RankMistral**.

| Model Name                                   | Hugging Face Identifier/Link                                 |
|----------------------------------------------|--------------------------------------------------------------|
| RankZephyr 7B V1 - Full - BF16               | [castorini/rank_zephyr_7b_v1_full](https://huggingface.co/castorini/rank_zephyr_7b_v1_full) |
| RankVicuna 7B V1                             | [castorini/rank_vicuna_7b_v1](https://huggingface.co/castorini/rank_vicuna_7b_v1) |
| RankVicuna 7B V1 - No Data Augmentation      | [castorini/rank_vicuna_7b_v1_noda](https://huggingface.co/castorini/rank_vicuna_7b_v1_noda) |
| RankMistral 7B V1                            | [AnonymousUser23456/RankMistral](https://huggingface.co/AnonymousUser23456/RankMistral) |

To run the experiments:

1. In the `rank_llm/src/rank_llm/rerank/rankllm.py` file:
   - Comment out lines 268 to 283.
   - Uncomment lines 303-306.
   - Comment line 309.
   
2. To save outputs for aggregation, uncomment lines 201 to 205 and specify the desired path.

3. For other _k_ evaluations, edit lines 117-130 in `rank_llm/src/rank_llm/retrieve_and_rerank.py`.

4. In the `run_example.bash` script, edit the model or dataset you want to use.

### Finetune your own models
To fine-tune your own LLMs, navigate to the `scripts` folder and modify the model name within the `run_train.sh` script. Once updated, execute the script using `bash run_train.sh`. 

We performed full fine-tuning using a single NVIDIA RTX A6000 GPU, which features 10,752 CUDA cores and 48 GB of VRAM.
