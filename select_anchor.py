from transformers import DebertaV2Tokenizer, DebertaV2Model
import torch
import json
from tqdm import tqdm
from pathlib import Path
import argparse
from sklearn.cluster import KMeans
import numpy as np

batch_size = 32

def get_first_k_tokens(text, k):
    """Extract first k tokens from text"""
    tokens = text.split()
    output = " ".join(tokens[:k])
    return output

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def extract_article(text):
    marker = "] description: "
    marker_pos = text.find(marker)
    
    if marker_pos == -1:
        raise ValueError()

    extracted_string = text[marker_pos + len(marker):]
    return extracted_string

parser = argparse.ArgumentParser("Anchor user selection", add_help=False)
parser.add_argument("--candidate_path", default="../data/movie_tagging/user_anchor_candidate.json", type=str, help="path to candidate user json file")
parser.add_argument("--task_name", default="movie_tagging", type=str, metavar="TASK", help="name of the task")
parser.add_argument("--k", default=50, type=int,help="number of selected anchor user")

args = parser.parse_args()

print("Loading data...")
with open(args.candidate_path, 'r') as f:
    anchor_candidate = json.load(f)

with open('./prompt/prompt.json', 'r') as f:
    prompt_template = json.load(f)

# Load DeBERTa model
print("Loading DeBERTa model...")
model_path = "./model/deberta-v3-large"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
model = DebertaV2Model.from_pretrained(model_path).cuda()

print("Generating user embeddings...")
all_user_emb = []
for user in tqdm(anchor_candidate, desc="Processing users"):

    history_embeddings_list = []

    visible_history_list = user['profile']
    for p in visible_history_list:
        for key, value in p.items():
            if isinstance(value, str):
                p[key] = get_first_k_tokens(value, 368)

    user_nl_history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]

    user_nl_history_list_batched = split_batch(user_nl_history_list, batch_size)

    for batch in tqdm(user_nl_history_list_batched, desc="Batching", leave=False):

        with torch.no_grad():
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            outputs = model(**inputs)

            last_hidden_states = outputs.last_hidden_state
            # Calculate attention mask
            attention_mask = inputs['attention_mask']

            # Expand attention mask for broadcasting
            attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()

            # Mean pooling
            masked_hidden_states = last_hidden_states * attention_mask
            summed = torch.sum(masked_hidden_states, 1)
            count = torch.clamp(attention_mask.sum(1), min=1e-9)
            mean_pooled = summed / count

        history_embeddings_list.append(mean_pooled.cpu())

    history_embedding_concat = torch.cat(history_embeddings_list, dim=0).cpu().mean(dim=0, keepdim=True)
    all_user_emb.append(history_embedding_concat)

all_user_emb = torch.cat(all_user_emb, dim=0)
print(f"Generated embeddings for {all_user_emb.size(0)} users")

# Output directory will be in data folder

print("Performing K-Means clustering...")
emb = all_user_emb.numpy()

k=args.k
kmeans = KMeans(n_clusters=k, random_state=0, max_iter=3000).fit(emb)
labels = kmeans.labels_

print("Selecting anchor users...")
selected_indices = []

for i in tqdm(range(k), desc="Selecting anchors"):
    cluster_indices = np.where(labels == i)[0]
    max_len = 0
    for idx in cluster_indices:
        if len(anchor_candidate[idx]['profile']) > max_len:
            max_len = len(anchor_candidate[idx]['profile'])
            selected_index = idx

    if max_len>10:
        selected_indices.append(selected_index)

print(f"Selected {len(selected_indices)} anchor users")

# Save anchor user data to jsonl file in the data folder
output_jsonl_path = f'./data/{args.task_name}/anchor_user_data.jsonl'
with open(output_jsonl_path, 'w') as f_out:
    for idx in selected_indices:
        user_data = anchor_candidate[idx]
        f_out.write(json.dumps(user_data) + '\n')

print(f"Saved anchor data to {output_jsonl_path}")
print('Done!')