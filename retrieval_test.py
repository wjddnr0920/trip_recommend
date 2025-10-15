import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torch.amp import autocast
import pickle

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
            return {"image": processed_image, "id": image_id}
        except FileNotFoundError:
            return None

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    images = torch.stack([d['image'] for d in batch])
    ids = [d['id'] for d in batch]
    return {"image": images, "id": ids}

def calculate_map(predictions, ground_truth):
    """mean Average Precision (mAP)을 계산하는 함수"""
    total_average_precision = 0.0
    # 유효한 쿼리(정답이 있는 쿼리)의 개수를 세기 위한 카운터
    valid_queries_count = 0
    
    for query_id, retrieved_ids in predictions.items():
        # 정답지가 없거나 비어있는 쿼리는 평가에서 제외
        if query_id not in ground_truth or not ground_truth[query_id]:
            continue

        # 유효한 쿼리이므로 카운터 증가
        valid_queries_count += 1
        
        relevant_ids = ground_truth[query_id]
        score, num_hits = 0.0, 0.0
        for i, p_id in enumerate(retrieved_ids):
            if p_id in relevant_ids:
                num_hits += 1.0
                precision_at_i = num_hits / (i + 1.0)
                score += precision_at_i
        
        # 각 쿼리의 AP는 해당 쿼리의 정답 개수로 나눔
        average_precision = score / len(relevant_ids)
        total_average_precision += average_precision
    
    # 분모를 전체 예측 수가 아닌, 유효한 쿼리의 수로 사용하여 GAP를 정확하게 계산
    if valid_queries_count == 0:
        return 0.0
    return total_average_precision / valid_queries_count

def perform_retrieval():
    with open('/home/workspace/config_custom.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() and config['system']['device'] == 'auto' else "cpu"
    print(f"Using device: {device}")
    
    use_amp = config['retrieval']['use_amp'] and device == 'cuda'
    print(f"Retrieval with AMP is {'ENABLED' if use_amp else 'DISABLED'}")

    print("Loading model and database...")
    processor = CLIPProcessor.from_pretrained(config['model']['model_id'], use_fast=True)
    model = CLIPModel.from_pretrained(config['model']['model_id']).to(device)
    finetuned_path = config['model'].get('finetuned_path')
    if finetuned_path and os.path.exists(finetuned_path):
        print(f"Loading fine-tuned weights from {finetuned_path}")
        model.load_state_dict(torch.load(finetuned_path))
    else:
        print("Using pre-trained weights.")
        
    model.eval()

    output_dir = config['paths']['output_dir']
    # 1. 통합된 인덱스 파일 로드
    index = faiss.read_index(os.path.join(output_dir, "image_features.index"))
    # 2. ID 매핑 파일 로드
    with open(os.path.join(output_dir, "id_map.pkl"), 'rb') as f:
        id_map = pickle.load(f)
    print(f"Loading complete. Index contains {index.ntotal} vectors.")

    solution_df = pd.read_csv(config['paths']['solution_csv_path'])
    test_df = solution_df[solution_df['Usage'] != 'Ignored'].copy()
    
    ground_truth = {row['id']: set(row['images'].split(' ')) for _, row in test_df.iterrows() if isinstance(row['images'], str)}

    test_dataset = TestDataset(df=test_df, image_dir=config['paths']['test_image_dir'], processor=processor)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config['retrieval']['batch_size'], shuffle=False,
        num_workers=config['system']['num_workers'], collate_fn=collate_fn_skip_none, pin_memory=True
    )
    
    # K 값을 Faiss 인덱스의 전체 벡터 수보다 크지 않도록 안전하게 조정
    num_neighbors = min(config['retrieval']['num_neighbors'], index.ntotal)
    
    all_test_ids, all_retrieved_int_ids = [], []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Performing retrieval"):
            if batch is None: continue
            images = batch['image'].to(device, non_blocking=True)
            test_ids = batch['id']
            
            with autocast(enabled=use_amp, device_type=device):
                query_embeddings = model.get_image_features(pixel_values=images)
                query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)

            # `search`는 이제 순차 인덱스가 아닌, '정수 ID'를 직접 반환
            _, neighbor_int_ids = index.search(query_embeddings.cpu().float().numpy(), num_neighbors)
            
            all_test_ids.extend(test_ids)
            all_retrieved_int_ids.extend(neighbor_int_ids)

    # 3. ID 맵을 사용하여 정수 ID를 원래의 문자열 ID로 변환
    predictions = {}
    for test_id, int_ids in zip(all_test_ids, all_retrieved_int_ids):
        # Faiss가 반환한 (잠재적 음수) int64 배열을 uint64로 재해석
        uint_ids = np.array(int_ids, dtype=np.uint64)
        
        predictions[test_id] = [id_map[i.item()] for i in uint_ids if i != -1]
        
    print("\nEvaluating performance...")
    map_score = calculate_map(predictions, ground_truth)
    print(f"mean Average Precision (mAP) @{num_neighbors}: {map_score:.4f}")

if __name__ == '__main__':
    perform_retrieval()
