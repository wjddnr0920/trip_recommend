import os
# DataLoader 멀티프로세싱과 Tokenizer 충돌 방지
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

class TestDataset(Dataset):
    """테스트 이미지를 로드하는 데이터셋 클래스"""
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
            # 파일이 없는 경우 None을 반환하여 collate_fn에서 처리
            return None

def collate_fn_skip_none(batch):
    """
    데이터 로딩 시 None인 샘플을 건너뛰고, 키를 사용해 명시적으로 데이터를 처리하는
    안정적인 collate 함수
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    
    # 'image'와 'id' 키를 직접 사용하여 데이터를 추출
    images = torch.stack([d['image'] for d in batch])
    ids = [d['id'] for d in batch]
    
    return {"image": images, "id": ids}

def calculate_gap(predictions, ground_truth):
    """Global Average Precision (GAP)을 계산하는 함수"""
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
    """메인 검색 작업을 수행하는 함수"""
    # 1. 설정 로드
    with open('/home/workspace/config_custom.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() and config['system']['device'] == 'auto' else "cpu"
    print(f"Using device: {device}")
    
    use_amp = config['retrieval']['use_amp'] and device == 'cuda'
    print(f"Retrieval with AMP is {'ENABLED' if use_amp else 'DISABLED'}")

    # 2. 모델, Faiss 인덱스, ID 맵핑 불러오기
    print("Loading model and database...")
    processor = CLIPProcessor.from_pretrained(config['model']['model_id'], use_fast=True)
    
    # 파인튜닝된 가중치를 반영하기 위해 전체 CLIPModel 로드
    model = CLIPModel.from_pretrained(config['model']['model_id']).to(device)
    finetuned_path = config['model'].get('finetuned_path') # .get()으로 안전하게 접근
    if finetuned_path and os.path.exists(finetuned_path):
        print(f"Loading fine-tuned weights from {finetuned_path}")
        model.load_state_dict(torch.load(finetuned_path))
    else:
        print("Fine-tuned path not specified or not found. Using pre-trained weights.")
        
    model.eval()

    output_dir = config['paths']['output_dir']
    index = faiss.read_index(os.path.join(output_dir, "image_features.index"))
    with open(os.path.join(output_dir, "image_ids.txt"), 'r') as f:
        index_image_ids = [line.strip() for line in f.readlines()]
    print("Loading complete.")

    # 3. Test 데이터 및 정답 데이터 준비
    solution_df = pd.read_csv(config['paths']['solution_csv_path'])
    test_df = solution_df[solution_df['Usage'] == 'Public'].copy()
    
    ground_truth = {
        row['id']: set(row['images'].split(' ')) 
        for _, row in test_df.iterrows() if isinstance(row['images'], str)
    }

    test_dataset = TestDataset(df=test_df, image_dir=config['paths']['test_image_dir'], processor=processor)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config['retrieval']['batch_size'], 
        shuffle=False,
        num_workers=config['system']['num_workers'],
        collate_fn=collate_fn_skip_none,
        pin_memory=True # non_blocking=True를 위해 필수
    )
    
    # 4. Test 이미지에 대해 검색 수행
    all_test_ids, all_retrieved_indices = [], []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Performing retrieval on test images"):
            if batch is None:
                continue

            images = batch['image'].to(device, non_blocking=True)
            test_ids = batch['id']
            
            with autocast(enabled=use_amp, device_type=device):
                # 프로젝션 레이어까지 거친 최종 임베딩 추출
                query_embeddings = model.get_image_features(pixel_values=images)
                # IndexFlatIP 사용을 위한 L2-정규화
                query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)

            # Faiss 검색
            _, neighbor_indices = index.search(query_embeddings.cpu().float().numpy(), config['retrieval']['num_neighbors'])
            
            all_test_ids.extend(test_ids)
            all_retrieved_indices.extend(neighbor_indices)

    # 5. 검색 결과를 이미지 ID로 변환
    predictions = {
        test_id: [index_image_ids[i] for i in indices]
        for test_id, indices in zip(all_test_ids, all_retrieved_indices)
    }
        
    # 6. 성능 평가 (GAP 계산)
    print("\nEvaluating performance...")
    gap_score = calculate_gap(predictions, ground_truth)
    print(f"Global Average Precision (GAP) @{config['retrieval']['num_neighbors']}: {gap_score:.4f}")

if __name__ == '__main__':
    # 필요한 파일 존재 여부 확인
    config_path = '/home/workspace/config_custom.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
    else:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        output_dir = cfg['paths']['output_dir']
        faiss_index_path = os.path.join(output_dir, "image_features.index")
        image_ids_path = os.path.join(output_dir, "image_ids.txt")

        if not all(os.path.exists(f) for f in [faiss_index_path, image_ids_path]):
            print("Error: Faiss index or ID file not found in the output directory.")
            print("Please run the embedding_DB.py script first to create the database.")
        else:
            perform_retrieval()

