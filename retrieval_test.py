import os
import yaml # YAML 라이브러리 임포트
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# --- Config Class Removed ---

class TestDataset(Dataset):
    def __init__(self, df, image_dir, processor):
        self.df = df
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['id']
        img_path = os.path.join(self.image_dir, image_id[0], image_id[1], image_id[2], f"{image_id}.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
            processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color='black')
            processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            image_id = f"not_found_{image_id}"
        return {"image": processed_image, "id": image_id}

def calculate_gap(predictions, ground_truth):
    total_average_precision = 0.0
    for query_id, retrieved_ids in predictions.items():
        if query_id not in ground_truth or len(ground_truth[query_id]) == 0:
            continue
        relevant_ids = ground_truth[query_id]
        score, num_hits = 0.0, 0.0
        for i, p_id in enumerate(retrieved_ids):
            if p_id in relevant_ids:
                num_hits += 1.0
                precision_at_i = num_hits / (i + 1.0)
                score += precision_at_i
        average_precision = score / len(relevant_ids)
        total_average_precision += average_precision
    return total_average_precision / len(predictions)

def perform_retrieval():
    # Load configuration from YAML file
    with open('/home/workspace/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Determine device
    if config['system']['device'] == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config['system']['device']
    print(f"Using device: {device}")

    print("Loading model and database...")
    model = CLIPModel.from_pretrained(config['model']['model_id']).vision_model.to(device)
    model.eval()

    output_dir = config['paths']['output_dir']
    index = faiss.read_index(os.path.join(output_dir, "image_features.index"))
    with open(os.path.join(output_dir, "image_ids.txt"), 'r') as f:
        index_image_ids = [line.strip() for line in f.readlines()]
    print("Loading complete.")

    solution_df = pd.read_csv(config['paths']['solution_csv_path'])
    test_df = solution_df[solution_df['Usage'] == 'Public'].copy()
    
    ground_truth = {}
    for _, row in test_df.iterrows():
        if isinstance(row['images'], str):
            ground_truth[row['id']] = set(row['images'].split(' '))

    processor = CLIPProcessor.from_pretrained(config['model']['model_id'], use_fast=True)
    test_dataset = TestDataset(df=test_df, image_dir=config['paths']['test_image_dir'], processor=processor)
    test_dataloader = DataLoader(test_dataset, batch_size=config['retrieval']['batch_size'], shuffle=False)
    
    all_test_ids, all_retrieved_indices = [], []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Performing retrieval on test images"):
            images = batch['image'].to(device)
            test_ids = batch['id']
            outputs = model(pixel_values=images)
            query_embeddings = F.normalize(outputs.pooler_output, p=2, dim=-1).cpu().numpy()
            _, neighbor_indices = index.search(query_embeddings, config['retrieval']['num_neighbors'])
            all_test_ids.extend(test_ids)
            all_retrieved_indices.extend(neighbor_indices)

    predictions = {
        test_id: [index_image_ids[i] for i in indices]
        for test_id, indices in zip(all_test_ids, all_retrieved_indices)
    }
        
    print("\nEvaluating performance...")
    gap_score = calculate_gap(predictions, ground_truth)
    print(f"Global Average Precision (GAP) @{config['retrieval']['num_neighbors']}: {gap_score:.4f}")

if __name__ == '__main__':
    perform_retrieval()
