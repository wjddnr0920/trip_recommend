import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import os
import faiss
import numpy as np

# --- Configuration ---
class Config:
    # 경로 설정
    DATA_DIR = '/home/workspace/data/GLDv2/test'
    TEST_IMAGE_DIR = os.path.join(DATA_DIR, 'image')      # Test 이미지 경로
    SOLUTION_CSV_PATH = os.path.join(DATA_DIR, 'retrieval_solution_v2.1.csv') # 정답 파일 경로
    
    # 이전에 생성한 DB 파일 경로
    DB_DIR = '/home/workspace/data/GLDv2/index/DB'
    FAISS_INDEX_PATH = os.path.join(DB_DIR, 'image_features.index')
    IMAGE_IDS_PATH = os.path.join(DB_DIR, 'image_ids.txt')

    # 모델 설정
    MODEL_NAME = 'openai/clip-vit-base-patch32'
    # 파인튜닝된 모델 가중치가 있다면 경로를 지정하세요.
    MODEL_PATH = None 
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    EMBEDDING_DIM = 768 # DB 생성 시 사용했던 차원과 동일해야 함

    # Retrieval 설정
    NUM_NEIGHBORS = 100 # 각 test 이미지 당 검색할 index 이미지 개수

# --- Test Dataset ---
class TestDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df
        self.image_dir = image_dir
        self.processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME, use_fast=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['id']
        
        # 이미지 경로 구성 (GLDv2 형식)
        img_path = os.path.join(self.image_dir, image_id[0], image_id[1], image_id[2], f"{image_id}.jpg")
        
        try:
            image = Image.open(img_path).convert("RGB")
            processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        except FileNotFoundError:
            # 파일을 못 찾으면 검은색 이미지로 대체하여 오류 방지
            image = Image.new('RGB', (224, 224), color='black')
            processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            image_id = f"not_found_{image_id}"

        return {"image": processed_image, "id": image_id}

# --- Evaluation Metric: Global Average Precision (GAP) ---
def calculate_gap(predictions, ground_truth):
    """
    predictions: {query_id: [retrieved_id_1, retrieved_id_2, ...]}
    ground_truth: {query_id: {relevant_id_1, relevant_id_2, ...}}
    """
    total_average_precision = 0.0
    
    for query_id, retrieved_ids in predictions.items():
        if query_id not in ground_truth or len(ground_truth[query_id]) == 0:
            continue

        relevant_ids = ground_truth[query_id]
        
        score = 0.0
        num_hits = 0.0
        
        for i, p_id in enumerate(retrieved_ids):
            if p_id in relevant_ids:
                num_hits += 1.0
                precision_at_i = num_hits / (i + 1.0)
                score += precision_at_i
        
        average_precision = score / len(relevant_ids)
        total_average_precision += average_precision
        
    return total_average_precision / len(predictions)

# --- Main Retrieval Function ---
def perform_retrieval():
    print(f"사용 디바이스: {Config.DEVICE}")

    # 1. 모델, Faiss 인덱스, ID 맵핑 불러오기
    print("모델 및 데이터베이스 로딩 중...")
    model = CLIPModel.from_pretrained(Config.MODEL_NAME).vision_model.to(Config.DEVICE)
    model.eval()

    index = faiss.read_index(Config.FAISS_INDEX_PATH)
    
    with open(Config.IMAGE_IDS_PATH, 'r') as f:
        index_image_ids = [line.strip() for line in f.readlines()]

    print("로딩 완료.")

    # 2. Test 데이터 및 정답 데이터 준비
    solution_df = pd.read_csv(Config.SOLUTION_CSV_PATH)
    # Private / Public 데이터셋 중 Public만 사용 (Usage 열 기준)
    test_df = solution_df[solution_df['Usage'] == 'Public'].copy()
    
    ground_truth = {}
    for _, row in test_df.iterrows():
        # images 열이 비어있지 않은 경우에만 처리
        if isinstance(row['images'], str):
            ground_truth[row['id']] = set(row['images'].split(' '))

    test_dataset = TestDataset(df=test_df, image_dir=Config.TEST_IMAGE_DIR)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 3. Test 이미지에 대해 검색 수행
    all_test_ids = []
    all_retrieved_indices = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test 이미지 검색 중"):
            images = batch['image'].to(Config.DEVICE)
            test_ids = batch['id']
            
            # 쿼리 임베딩 생성
            outputs = model(pixel_values=images)
            query_embeddings = outputs.pooler_output
            query_embeddings = F.normalize(query_embeddings, p=2, dim=-1).cpu().numpy()
            
            # Faiss 검색
            _, neighbor_indices = index.search(query_embeddings, Config.NUM_NEIGHBORS)
            
            all_test_ids.extend(test_ids)
            all_retrieved_indices.extend(neighbor_indices)

    # 4. 검색 결과를 이미지 ID로 변환
    predictions = {}
    for test_id, indices in zip(all_test_ids, all_retrieved_indices):
        retrieved_ids = [index_image_ids[i] for i in indices]
        predictions[test_id] = retrieved_ids
        
    # 5. 성능 평가 (GAP 계산)
    print("\n성능 평가 중...")
    gap_score = calculate_gap(predictions, ground_truth)
    print(f"Global Average Precision (GAP) @{Config.NUM_NEIGHBORS}: {gap_score:.4f}")


if __name__ == '__main__':
    # 필요한 파일 존재 여부 확인
    required_files = [Config.FAISS_INDEX_PATH, Config.IMAGE_IDS_PATH, Config.SOLUTION_CSV_PATH]
    if not all(os.path.exists(f) for f in required_files):
        print("에러: Faiss 인덱스, ID 파일, 또는 정답 CSV 파일이 존재하지 않습니다.")
        print("먼저 embedding_DB.py를 실행하여 데이터베이스를 생성했는지 확인해주세요.")
    else:
        perform_retrieval()


### 코드 사용 방법
"""
1.  **파일 위치**: 이 `retrieval.py` 파일을 이전에 작업했던 `embedding_DB.py`와 같은 위치에 저장합니다.
2.  **데이터 준비**:
    * `./test/` 디렉토리에 test 이미지들이 있어야 합니다.
    * `./retrieval_solution_v2.1.csv` 파일이 있어야 합니다.
    * `./output/` 디렉토리에 `embedding_DB.py` 실행 결과물인 `image_features.index`와 `image_ids.txt` 파일이 있어야 합니다.
3.  **실행**: 터미널에서 다음 명령어를 입력하여 코드를 실행합니다.
    ```bash
    python retrieval.py
"""
