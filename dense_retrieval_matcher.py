import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple, Any, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DenseRetrievalMatcher:
    def __init__(self, model_name: str = "./model/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self.few_shot_embeddings = None
        self.anchor_embeddings = None
        self.few_shot_data = None
        self.anchor_data = None
        
    def load_model(self):
        logger.info(f"Loading BGE model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_few_shot_data(self, filepath: str):
        logger.info(f"Loading few-shot data from: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.few_shot_data = json.load(f)
            logger.info(f"Loaded {len(self.few_shot_data)} few-shot users")
        except Exception as e:
            logger.error(f"Failed to load few-shot data: {e}")
            raise
    
    def load_anchor_data(self, filepath: str):
        logger.info(f"Loading anchor data from: {filepath}")
        try:
            self.anchor_data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.anchor_data.append(json.loads(line))
            logger.info(f"Loaded {len(self.anchor_data)} anchor users")
        except Exception as e:
            logger.error(f"Failed to load anchor data: {e}")
            raise
    
    def extract_user_text(self, user_data: Dict, task_name: str) -> str:
        texts = []
        

        if task_name == "product_rating":
            if 'profile' in user_data:
                profile = user_data['profile']
                for review in profile:
                    if 'text' in review:
                        texts.append(review['text'])
        
        elif task_name == "scholarly_title":
            if 'profile' in user_data:
                profile = user_data['profile']
                for item in profile:
                    title = item.get('title', '')
                    abstract = item.get('abstract', '')
                    if title or abstract:
                        texts.append(f"{title} {abstract}".strip())
                        
        elif task_name == "news_headline":
            if 'profile' in user_data:
                profile = user_data['profile']
                for item in profile:
                    title = item.get('title', '')
                    text = item.get('text', '')
                    if title or text:

                        texts.append(f"{title} {text}".strip())
        
        elif task_name == "citation":
            if 'profile' in user_data:
                profile = user_data['profile']
                for item in profile:
                    title = item.get('title', '')
                    abstract = item.get('abstract', '')
                    citation = item.get('citation', '')
                    if title or abstract or citation:
                        texts.append(f"{title} {abstract} {citation}".strip())

        elif task_name == "movie_tagging":
            if 'profile' in user_data:
                profile = user_data['profile']
                for item in profile:
                    tag = item.get('tag', '')
                    description = item.get('description', '')
                    if tag or description:
                        texts.append(f"{tag} {description}".strip())
        else:
            logger.warning(f"Unknown task name: {task_name}. Using default text extraction logic.")

            if 'profile' in user_data:
                profile = user_data['profile']
                for item in profile:
                    for key, value in item.items():
                        if isinstance(value, str):
                            texts.append(value)


        combined_text = " ".join(texts)
        return combined_text.strip()
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=True)
            embeddings.append(batch_embeddings.cpu().numpy())
        

        all_embeddings = np.vstack(embeddings)
        logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
        return all_embeddings
    
    def prepare_data(self, task_name: str):
        if self.few_shot_data is None or self.anchor_data is None:
            raise ValueError("Data not loaded. Please load data first.")
        
        logger.info(f"Preparing few-shot data for task: {task_name}")
        few_shot_texts = [self.extract_user_text(user, task_name) for user in self.few_shot_data]
        
        logger.info(f"Preparing anchor data for task: {task_name}")
        anchor_texts = [self.extract_user_text(user, task_name) for user in self.anchor_data]
        

        self.few_shot_embeddings = self.generate_embeddings(few_shot_texts)
        self.anchor_embeddings = self.generate_embeddings(anchor_texts)
    
    def find_matches(self, top_k: int = 5) -> List[Dict]:
        if self.few_shot_embeddings is None or self.anchor_embeddings is None:
            raise ValueError("Embeddings not prepared. Please call prepare_data() first.")
        
        if self.few_shot_data is None or self.anchor_data is None:
            raise ValueError("Data not loaded. Please load data first.")
            
        logger.info(f"Finding top-{top_k} matches for {len(self.few_shot_data)} few-shot users")
        

        similarity_matrix = cosine_similarity(self.few_shot_embeddings, self.anchor_embeddings)
        
        results = []
        for i, few_shot_user in enumerate(tqdm(self.few_shot_data, desc="Finding matches")):

            similarities = similarity_matrix[i]
            

            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            matches = []
            for j, anchor_idx in enumerate(top_indices):
                match = {
                    'rank': j + 1,
                    'anchor_user_id': self.anchor_data[anchor_idx]['user_id'],
                    'similarity_score': float(similarities[anchor_idx])
                }
                matches.append(match)
            
            result = {
                'few_shot_user_id': few_shot_user['user_id'],
                'matches': matches
            }
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        logger.info(f"Saving results to: {output_path}")
        

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info("Results saved successfully")
    
    def run_matching(self, task_name: str, top_k: int = 5):

        few_shot_path = f"./data/{task_name}/test_100/test_100.json"
        anchor_path = f"./data/{task_name}/anchor_user_data.jsonl"
        output_path = f"./dense_retrieval_results/{task_name}/matches_results.json"
        

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        

        self.load_model()
        

        self.load_few_shot_data(few_shot_path)
        self.load_anchor_data(anchor_path)
        

        logger.info("-------------------- Data Statistics --------------------")
        total_few_shot_users = len(self.few_shot_data) if self.few_shot_data else 0
        logger.info(f"Task Name: {task_name}")
        logger.info(f"Total few-shot users to match: {total_few_shot_users}")
        logger.info(f"Total anchor users in pool: {len(self.anchor_data) if self.anchor_data else 0}")

        if total_few_shot_users > 0:
            profile_lengths = [len(user.get('profile', [])) for user in self.few_shot_data]
            logger.info("Few-shot user profile length stats:")
            logger.info(f"  - Min: {min(profile_lengths)}")
            logger.info(f"  - Max: {max(profile_lengths)}")
            logger.info(f"  - Average: {np.mean(profile_lengths):.2f}")
            
        if self.anchor_data:
            anchor_profile_lengths = [len(user.get('profile', [])) for user in self.anchor_data]
            if anchor_profile_lengths:
                logger.info("Anchor user profile length stats:")
                logger.info(f"  - Min: {min(anchor_profile_lengths)}")
                logger.info(f"  - Max: {max(anchor_profile_lengths)}")
                logger.info(f"  - Average: {np.mean(anchor_profile_lengths):.2f}")
                
        logger.info("-------------------------------------------------------")
        

        self.prepare_data(task_name)
        

        results = self.find_matches(top_k)
        

        self.save_results(results, output_path)
        

        logger.info(f"Matching completed for task '{task_name}'!")
        logger.info(f"Top-K matches per user: {top_k}")
        logger.info(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Dense Retrieval Matcher for Few-shot Learning")
    parser.add_argument("--task_name", type=str, required=True,
                       help="Name of the task (e.g., product_rating, movie_tagging)")
    parser.add_argument("--top_k", type=int, default=2,
                       help="Number of top similar anchors to return")
    parser.add_argument("--model_name", type=str, default="./model/bge-small-en-v1.5",
                       help="BGE model name or local path")
    
    args = parser.parse_args()
    
    matcher = DenseRetrievalMatcher(model_name=args.model_name)
    matcher.run_matching(task_name=args.task_name, top_k=args.top_k)

if __name__ == '__main__':
    main() 