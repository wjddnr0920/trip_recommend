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
# --- Config Class Removed ---

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
    print(f"Using device: {device}")
    
    use_amp = config['retrieval']['use_amp'] and device == 'cuda'
    print(f"Inference with AMP is {'ENABLED' if use_amp else 'DISABLED'}")

    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model and processor...")
    processor = CLIPProcessor.from_pretrained(config['model']['model_id'], use_fast=True)
    model = CLIPModel.from_pretrained(config['model']['model_id']).to(device)
    
    finetuned_path = config['model']['finetuned_path']
    if finetuned_path and os.path.exists(finetuned_path):
        print(f"Loading fine-tuned model from: {finetuned_path}")
        model.load_state_dict(torch.load(finetuned_path))
    else:
        print("Fine-tuned model not found. Using pre-trained weights.")
        
    model.eval()

    dataset = IndexDataset(
        csv_path=config['paths']['index_csv_path'],
        image_dir=config['paths']['index_image_dir'],
        processor=processor
    )

    # 커스텀 collate_fn을 DataLoader에 적용
    dataloader = DataLoader(
        dataset, 
        batch_size=config['retrieval']['batch_size'], 
        shuffle=False,
        collate_fn=collate_fn_skip_none
    )

    # 모델에서 임베딩 차원을 동적으로 가져옴
    embedding_dim = model.config.projection_dim
    print(f"Embedding dimension detected from model config: {embedding_dim}")
    
    # Faiss 인덱스를 먼저 초기화
    index = faiss.IndexFlatIP(embedding_dim)
    all_image_ids = []
    
    processed_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Creating and adding embeddings to Faiss index"):
            # collate_fn에 의해 배치가 비어있을 수 있음
            if batch is None:
                continue

            images = batch['image'].to(device)
            image_ids = batch['id']
            
            with autocast(enabled=use_amp, device_type=device):
                embeddings = model.get_image_features(pixel_values=images)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            # 임베딩을 메모리에 쌓아두지 않고, 배치 단위로 Faiss 인덱스에 바로 추가
            index.add(embeddings.cpu().float().numpy())
            all_image_ids.extend(image_ids)
            processed_count += len(image_ids)

    print(f"Successfully processed and indexed {processed_count} images.")
    print(f"Total vectors in Faiss index: {index.ntotal}")

    faiss.write_index(index, os.path.join(output_dir, "image_features.index"))
    with open(os.path.join(output_dir, "image_ids.txt"), "w") as f:
        for image_id in all_image_ids:
            f.write(f"{image_id}\n")
            
    print(f"Faiss index and ID list saved to '{output_dir}'")

if __name__ == '__main__':
    create_database()
