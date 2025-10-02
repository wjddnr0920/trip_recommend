import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel as HuggingFaceCLIPModel
from tqdm import tqdm
import os
import faiss
import numpy as np

# --- Configuration ---
class Config:
    # 경로 설정
    DATA_DIR = '/home/workspace/data/GLDv2/index' # index 데이터 최상위 디렉토리
    INDEX_IMAGE_DIR = os.path.join(DATA_DIR, 'image') # index 이미지 경로
    INDEX_CSV_PATH = os.path.join(DATA_DIR, 'index.csv') # index.csv 파일 경로
    OUTPUT_DIR = '/home/workspace/data/GLDv2/index/DB' # 결과물 저장 경로
    
    # 모델 설정
    MODEL_NAME = 'openai/clip-vit-base-patch32'
    # 만약 파인튜닝된 모델 가중치가 있다면 경로를 지정하세요.
    # 예: MODEL_PATH = './output/best_model.pth' 
    MODEL_PATH = None 
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64 # 추론 시에는 더 큰 배치 사이즈 사용 가능
    EMBEDDING_DIM = 768 # CLIP-base 모델의 임베딩 차원

# --- Dataset for Index Images ---
class IndexDataset(Dataset):
    def __init__(self, csv_path, image_dir):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME, use_fast=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['id']
        
        # 이미지 경로 구성 (GLDv2 형식: abc... -> a/b/c/abc....jpg)
        img_path = os.path.join(self.image_dir, image_id[0], image_id[1], image_id[2], f"{image_id}.jpg")
        
        try:
            image = Image.open(img_path).convert("RGB")
            processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        except FileNotFoundError:
            # 파일을 찾지 못하면 빈 이미지를 반환하여 중단을 막습니다.
            image = Image.new('RGB', (224, 224), color='black')
            processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            image_id = "file_not_found"

        return {"image": processed_image, "id": image_id}

# --- 이미지 특징 추출 모델 ---
# 학습 코드의 모델 정의를 재사용하거나, 사전 학습된 CLIP 모델을 바로 사용합니다.
class ImageEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.vision_model = HuggingFaceCLIPModel.from_pretrained(model_name).vision_model
        self.image_projection = nn.Linear(self.vision_model.config.hidden_size, Config.EMBEDDING_DIM)
        
    def forward(self, pixel_values):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embed = vision_outputs.pooler_output
        projected_embed = self.image_projection(image_embed)
        return F.normalize(projected_embed, p=2, dim=-1)

# --- Main Execution ---
def create_database():
    print(f"사용 디바이스: {Config.DEVICE}")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # 1. 모델 불러오기
    # 사전 학습된 CLIP의 이미지 인코더 부분만 사용
    model = HuggingFaceCLIPModel.from_pretrained(Config.MODEL_NAME).vision_model.to(Config.DEVICE)
    # 만약 파인튜닝된 모델을 사용한다면 아래와 같이 불러옵니다.
    # model = ImageEncoder(Config.MODEL_NAME)
    # model.load_state_dict(torch.load(Config.MODEL_PATH))
    # model = model.to(Config.DEVICE)
    model.eval() # 추론 모드로 설정

    # 2. 데이터셋 및 데이터로더 준비
    dataset = IndexDataset(csv_path=Config.INDEX_CSV_PATH, image_dir=Config.INDEX_IMAGE_DIR)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 3. 임베딩 생성 (핵심 로직)
    all_image_embeddings = []
    all_image_ids = []

    # `torch.no_grad()`: 추론 시에는 그래디언트 계산이 필요 없으므로 메모리 사용량과 계산 속도를 향상시킵니다.
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="인덱스 이미지 임베딩 생성 중"):
            images = batch['image'].to(Config.DEVICE)
            image_ids = batch['id']
            
            # 모델을 통해 이미지 임베딩(피처) 추출
            # 사전 학습된 모델을 직접 사용할 경우 pooler_output을 사용
            outputs = model(pixel_values=images)
            embeddings = outputs.pooler_output
            
            # L2 정규화 (유사도 계산 시 성능 향상)
            embeddings = F.normalize(embeddings, p=2, dim=-1)

            all_image_embeddings.append(embeddings.cpu()) # GPU 메모리 절약을 위해 CPU로 이동
            all_image_ids.extend(image_ids)

    # 리스트에 저장된 텐서들을 하나의 텐서로 결합
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0).numpy()

    print(f"총 {len(all_image_ids)}개의 이미지 처리 완료.")
    print(f"생성된 임베딩 행렬 크기: {all_image_embeddings.shape}") # (이미지 개수, 임베딩 차원)

    # 4. Faiss 인덱스 생성 및 저장
    index = faiss.IndexFlatIP(Config.EMBEDDING_DIM) # 내적(Inner Product) 기반 인덱스
    # L2 정규화된 벡터의 내적은 코사인 유사도와 비례하므로 IndexFlatIP가 적합합니다.
    # faiss.IndexFlatL2(Config.EMBEDDING_DIM) # L2 거리 기반 인덱스도 사용 가능합니다.
    
    index.add(all_image_embeddings)
    faiss.write_index(index, os.path.join(Config.OUTPUT_DIR, "image_features.index"))
    
    # 5. 이미지 ID 리스트 저장
    # Faiss 인덱스는 숫자 인덱스만 저장하므로, 원래 이미지 ID와 매핑할 정보가 필요합니다.
    with open(os.path.join(Config.OUTPUT_DIR, "image_ids.txt"), "w") as f:
        for image_id in all_image_ids:
            f.write(f"{image_id}\n")
            
    print(f"Faiss 인덱스와 ID 리스트가 '{Config.OUTPUT_DIR}'에 성공적으로 저장되었습니다.")

if __name__ == '__main__':
    create_database()