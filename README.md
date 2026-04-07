# MTA: A Merge-then-Adapt Framework for Personalized Large Language Model

**MTA** is a framework for Personalized Large Language Models (PLLMs) that addresses two key challenges in user-centric LLM adaptation: (1) storage costs that scale linearly with the number of users, and (2) suboptimal performance for users with sparse data. MTA comprises three stages:

1. **Meta-LoRA Bank Construction** — Select anchor users and pre-train meta-personalization traits in meta-LoRA modules.
2. **Adaptive LoRA Fusion** — Retrieve and dynamically merge the most relevant anchor meta-LoRAs to synthesize a user-specific adapter, eliminating per-user storage.
3. **LoRA Stacking** — Apply an additional ultra-low-rank LoRA on top of the merged LoRA for effective few-shot personalization.

Experiments on the [LaMP](https://arxiv.org/abs/2304.11406) benchmark demonstrate that MTA outperforms existing SOTA methods across multiple tasks.

**Paper:** [arXiv:2511.20072](https://arxiv.org/abs/2511.20072)

## Environment Setup

Create and activate a new conda environment named MTA:

```bash
conda create -n MTA python=3.10
conda activate MTA
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** LlamaFactory will be installed separately in the next step.

### Model Download

Download the required models to the `./model` directory:

```bash
# Create model directory
mkdir -p ./model

# Download BGE embedding model for dense retrieval
git clone https://huggingface.co/BAAI/bge-small-en-v1.5 ./model/bge-small-en-v1.5

# Download DeBERTa v3 Large model
git clone https://huggingface.co/microsoft/deberta-v3-large ./model/deberta-v3-large

# Download Llama 3 8B Instruct model
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct ./model/Meta-Llama-3-8B-Instruct
```

**Alternative download with huggingface-hub:**

```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Download models using huggingface CLI
huggingface-cli download BAAI/bge-small-en-v1.5 --local-dir ./model/bge-small-en-v1.5
huggingface-cli download microsoft/deberta-v3-large --local-dir ./model/deberta-v3-large
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./model/Meta-Llama-3-8B-Instruct
```

## Dataset

We use public available data from the [LaMP](https://arxiv.org/abs/2304.11406) benchmark for anchor selection. The dataset processing approach follows the same methodology as [Per-Pcs](https://arxiv.org/abs/2406.10471).

The processed datasets for anchor selection and training should be placed under the `./data` directory with the following structure:

```
./data/
├── citation/
├── movie_tagging/
├── news_headline/
├── product_rating/
└── scholarly_title/
```

Each task directory contains:
- `user_anchor_candidate.json`: Candidate users for anchor selection
- Test data for evaluation

## Usage

### Select Anchor Users
```bash
python select_anchor.py --candidate_path <data_path> --task_name <task> --k <num_anchors>
```

**Example:**
```bash
python select_anchor.py --candidate_path ./data/movie_tagging/user_anchor_candidate.json --task_name movie_tagging --k 50
```

**Parameters:**
- `--candidate_path`: Path to candidate user JSON file
- `--task_name`: Task name (movie_tagging, news_headline, citation, etc.)
- `--k`: Number of anchor users to select

### Train Meta-LoRA Bank

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && git checkout v0.9.2 && pip install -e . && cd ..
python prepare_meta_lora.py --task_name <task> --top_k 1
bash run_<task>_k1_training.sh
```

**Example:**
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && git checkout v0.9.2 && pip install -e . && cd ..
python prepare_meta_lora.py --task_name movie_tagging --top_k 1
bash run_movie_tagging_k1_training.sh
```

**Parameters:**
- `--task_name`: Task name (same as step 1)
- `--top_k`: Number of history items for BM25 retrieval (default: 1)

### Dense Retrieval Matching
```bash
python dense_retrieval_matcher.py --task_name <task> --top_k <num_matches>
```

**Example:**
```bash
python dense_retrieval_matcher.py --task_name movie_tagging --top_k 2
```

**Parameters:**
- `--task_name`: Task name (movie_tagging, citation, news_headline, product_rating, scholarly_title)
- `--top_k`: Number of top similar anchor users to return (default: 2)


**Output:** Results saved to `./dense_retrieval_results/{task_name}/matches_results.json`

### Complete MTA Experiment

After completing the above three steps, you can run the complete MTA experiment using the unified script:

```bash
bash run_experiment.sh <task_name> <num_users>
```

**Example:**
```bash
bash run_experiment.sh citation 10
```

**Parameters:**
- `<task_name>`: Task name (movie_tagging, citation, news_headline, product_rating, scholarly_title)
- `<num_users>`: Number of few-shot users to process


**Output Structure:**
```
./merge_lora/hybrid_{task_name}/           # Fused LoRA models
├── merge_lora_{user_id}/
└── ...

./adapt_lora/hybrid_{task_name}/           # Adapted stacked LoRA models  
├── adapt_lora_{user_id}/
└── ...

./final_eval_results/hybrid_{task_name}/   # Evaluation results
├── results_{user_id}_alpha_{weight}.json
├── final_aggregated_results_{task_name}.json
└── ...
```

## Citation

```bibtex
@misc{li2025mtamergethenadaptframeworkpersonalized,
      title={MTA: A Merge-then-Adapt Framework for Personalized Large Language Model},
      author={Xiaopeng Li and Yuanjin Zheng and Wanyu Wang and wenlin zhang and Pengyue Jia and Yiqi Wang and Maolin Wang and Xuetao Wei and Xiangyu Zhao},
      year={2025},
      eprint={2511.20072},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.20072},
}
```