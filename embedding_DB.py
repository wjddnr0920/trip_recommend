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
from torch.amp import autocast
import pickle

class IndexDataset(Dataset):
    def __init__(self, csv_path, image_dir, processor):
        self.df = pd.read_csv(csv_path)
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
        
# --- 추가한 함수 ---
def collate_fn_skip_none(batch):
    """
    데이터 로딩 시 None인 샘플을 건너뛰고, 키를 사용해 명시적으로 데이터를 처리하는
    안정적인 collate 함수
    """
    # 유효한 샘플(딕셔너리)만 필터링합니다.
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    
    # 'image'와 'id' 키를 직접 사용하여 데이터를 추출합니다.
    # 이렇게 하면 __getitem__에서 딕셔너리 순서가 바뀌어도 안전합니다.
    images = torch.stack([d['image'] for d in batch])
    ids = [d['id'] for d in batch]
    
    return {"image": images, "id": ids}

def create_database():
    with open('/home/workspace/config_custom.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() and config['system']['device'] == 'auto' else "cpu"
    use_amp = config['retrieval']['use_amp'] and device == 'cuda'
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model and processor...")
    processor = CLIPProcessor.from_pretrained(config['model']['model_id'], use_fast=True)
    model = CLIPModel.from_pretrained(config['model']['model_id']).to(device)
    
    finetuned_path = config['model'].get('finetuned_path')
    if finetuned_path and os.path.exists(finetuned_path):
        model.load_state_dict(torch.load(finetuned_path))
        print(f"Loaded fine-tuned model from: {finetuned_path}")
    else:
        print("Using pre-trained weights.")
    model.eval()

    dataset = IndexDataset(csv_path=config['paths']['index_csv_path'], image_dir=config['paths']['index_image_dir'], processor=processor)
    dataloader = DataLoader(dataset, batch_size=config['retrieval']['batch_size'], shuffle=False, collate_fn=collate_fn_skip_none)

    embedding_dim = model.config.projection_dim
    print(f"Embedding dimension: {embedding_dim}")

    # 1. 기본 인덱스(내적 유사도)를 생성합니다.
    base_index = faiss.IndexFlatIP(embedding_dim)
    # 2. IndexIDMap으로 기본 인덱스를 감싸서, 커스텀 ID를 사용할 수 있도록 합니다.
    index = faiss.IndexIDMap(base_index)
    
    id_map = {} # {정수 ID: 문자열 ID}를 저장할 딕셔너리
    processed_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Creating and adding embeddings to Faiss index"):
            if batch is None: continue

            images = batch['image'].to(device, non_blocking=True)
            str_image_ids = batch['id']
            
            with autocast(enabled=use_amp, device_type=device):
                embeddings = model.get_image_features(pixel_values=images)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            # 16진수 문자열 ID를 64비트 정수 ID로 변환
            int_image_ids = np.array([int(s_id, 16) for s_id in str_image_ids]).astype('uint64')

            # `add` 대신 `add_with_ids`를 사용하여 (벡터, ID) 쌍을 인덱스에 추가
            index.add_with_ids(embeddings.cpu().float().numpy(), int_image_ids)
            
            for str_id, int_id in zip(str_image_ids, int_image_ids):
                id_map[int_id.item()] = str_id # .item()으로 순수 파이썬 int로 변환
                
            processed_count += len(str_image_ids)

    print(f"Successfully processed and indexed {processed_count} images.")
    
    # 3. 하나의 인덱스 파일과 ID 맵 파일을 저장합니다.
    faiss.write_index(index, os.path.join(output_dir, "image_features.index"))
    with open(os.path.join(output_dir, "id_map.pkl"), "wb") as f:
        pickle.dump(id_map, f)
            
    print(f"Faiss index (with IDs) and ID map saved to '{output_dir}'")

if __name__ == '__main__':
    create_database()
