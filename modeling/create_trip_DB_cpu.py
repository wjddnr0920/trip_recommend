import os
import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import faiss
import numpy as np
from transformers import AutoProcessor, AutoModel
import pickle

# PyTorch Dataset 클래스 (DALI 대체)
class CustomDataset(Dataset):
    """
    DB 생성을 위한 커스텀 데이터셋 (PyTorch DataLoader용)
    """
    def __init__(self, metadata_path, img_root, path_col, id_col, processor):
        self.processor = processor
        self.valid_data = [] 
        
        print("Filtering valid image paths (CPU mode)...")
        df = pd.read_csv(metadata_path)
        
        # 유효한 이미지 경로 준비
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                relative_path = str(row[path_col]).strip()
                full_path = os.path.join(img_root, relative_path)
                
                if os.path.exists(full_path):
                    self.valid_data.append({
                        "path": full_path,
                        "id": str(row[id_col]).strip()
                    })
            except Exception:
                continue

        if not self.valid_data:
            print("Error: No valid images found! Check your CSV path and column names.")
            
        print(f"Found {len(self.valid_data)} valid images.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        img_path = item["path"]
        str_id = item["id"]
        
        try:
            # 이미지 로드(PIL)
            image = Image.open(img_path).convert("RGB")
            
            # AutoProcessor로 CLIP/SigLIP 여부에 따라 자동으로 리사이즈, 크롭, 정규화 수행
            inputs = self.processor(images=image, return_tensors="pt")
            
            # (1, C, H, W) -> (C, H, W) 차원 축소
            pixel_values = inputs['pixel_values'].squeeze(0)
            
            return {"image": pixel_values, "id": str_id}
        
        except Exception as e:
            print(f"Warning: Error reading {img_path}: {e}")
            return None

def collate_fn_skip_none(batch):
    """
    손상된 이미지(None)를 건너뛰고 배치를 구성하는 함수
    """

    # 배치 내의 None 데이터 필터링
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    
    # 이미지 텐서를 배치 단위로 스택 (C, H, W) -> (B, C, H, W)
    images = torch.stack([d['image'] for d in batch])
    ids = [d['id'] for d in batch]
    
    return {"image": images, "id": ids}

def create_database():
    with open('/home/workspace/configs/trip_config_custom.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # CPU 설정
    device = "cpu"
    print(f"Running on: {device}")
    
    # CPU에서는 AMP 비활성화
    print("AMP is DISABLED (CPU mode).")

    # 모델 및 프로세서 로드
    model_id = config['model']['model_id']
    print(f"Loading model and processor for: {model_id}")
    
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id).to(device)
    
    # 파인튜닝된 모델이 있다면 불러오기 (CPU 매핑)
    finetuned_path = config['model'].get('finetuned_path')
    if finetuned_path and os.path.exists(finetuned_path):
        model.load_state_dict(torch.load(finetuned_path, map_location=torch.device('cpu')))
        print(f"Loaded fine-tuned weights from: {finetuned_path}")
    
    model.eval()
    
    # Dataset 및 DataLoader 준비
    print("Preparing DataLoader...")
    dataset = CustomDataset(
        metadata_path=config['paths']['custom_metadata_csv'],
        img_root=config['paths'].get('custom_image_root', ''),
        path_col=config['custom_dataset_columns']['image_path_column'],
        id_col=config['custom_dataset_columns']['unique_id_column'],
        processor=processor
    )
    
    # pin_memory=False (CPU에서는 불필요)
    dataloader = DataLoader(
        dataset,
        batch_size=config['retrieval']['batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        collate_fn=collate_fn_skip_none
    )
    
    # CLIP 모델의 경우
    if hasattr(model.config, "projection_dim"):
        embedding_dim = model.config.projection_dim
        print(f"Embedding dimension (projection_dim) detected: {embedding_dim}")
    # SigLIP 시리즈 모델의 경우
    elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "projection_size"):
        embedding_dim = model.config.text_config.projection_size
        print(f"Embedding dimension (text_config.projection_size) detected: {embedding_dim}")
    else:
        # 안전하게 에러 발생
        raise AttributeError(f"Could not automatically determine embedding dimension for model {model_id}.")

    base_index = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIDMap(base_index)
    id_map = {} 
    current_index_counter = 0 
    
    with torch.no_grad():
        # CPU 모드이므로 autocast 제거
        for batch in tqdm(dataloader, desc="Creating embeddings (CPU)"):
            if batch is None: continue
            
            images = batch['image'].to(device) # 이미 CPU에 있지만 명시적으로 이동
            str_image_ids = batch['id']
            
            embeddings = model.get_image_features(pixel_values=images)
            embeddings = F.normalize(embeddings, p=2, dim=-1)

            '''
            DALI파이프라인을 사용하지 않으므로 이미지마다 고유 인덱스가 없음
            따라서 현재까지 처리한 이미지 수를 기준으로 정수 ID를 생성
            배치를 사용하기 때문에 고유 ID를 가질 수 있도록 현재 인덱스 카운터를 활용
            '''
            # 0부터 시작하는 순차적인 정수 ID 생성 (uint64)          
            num_in_batch = len(str_image_ids)
            int_image_ids = np.arange(current_index_counter, current_index_counter + num_in_batch, dtype=np.uint64)

            # Faiss에 추가
            index.add_with_ids(embeddings.numpy(), int_image_ids)
            
            # {정수 ID : 문자열 ID} 매핑 저장
            for i, str_id in enumerate(str_image_ids):
                id_map[int_image_ids[i].item()] = str_id
            
            current_index_counter += num_in_batch
                
    print(f"Successfully processed and indexed {index.ntotal} images.")
    
    # Faiss 인덱스 파일(.index)과 ID 매핑 파일(.pkl) 저장
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "image_features.index"))
    with open(os.path.join(output_dir, "id_map.pkl"), "wb") as f:
        pickle.dump(id_map, f)
    print(f"Faiss index and ID map saved to '{output_dir}'")

if __name__ == '__main__':
    create_database()