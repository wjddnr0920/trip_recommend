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
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color='black')
            processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            image_id = "file_not_found"
        return {"image": processed_image, "id": image_id}

def create_database():
    # Load configuration from YAML file
    with open('/home/workspace/config_custom.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Determine device
    if config['system']['device'] == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config['system']['device']
    print(f"Using device: {device}")
    
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model and processor...")
    processor = CLIPProcessor.from_pretrained(config['model']['model_id'], use_fast=True)

    finetuned_path = config['model']['finetuned_path']
    
    if finetuned_path and os.path.exists(finetuned_path):
        print(f"Loading fine-tuned model from: {finetuned_path}")
        # 1. 파인튜닝된 가중치를 로드하기 위해 전체 CLIPModel을 먼저 생성합니다.
        model_to_load = CLIPModel.from_pretrained(config['model']['model_id'])
        # 2. 전체 모델에 state_dict를 로드합니다. (키 이름이 일치하여 성공)
        model_to_load.load_state_dict(torch.load(finetuned_path))
        # 3. 필요한 vision_model 부분만 추출합니다.
        model = model_to_load.vision_model.to(device)
    else:
        print("Fine-tuned model not found or path not specified. Using pre-trained weights.")
        # 기존 방식: 사전 학습된 vision_model을 직접 로드합니다.
        model = CLIPModel.from_pretrained(config['model']['model_id']).vision_model.to(device)
        
    model.eval()

    dataset = IndexDataset(
        csv_path=config['paths']['index_csv_path'],
        image_dir=config['paths']['index_image_dir'],
        processor=processor
    )
    dataloader = DataLoader(dataset, batch_size=config['retrieval']['batch_size'], shuffle=False)

    all_image_embeddings = []
    all_image_ids = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Creating index image embeddings"):
            images = batch['image'].to(device)
            image_ids = batch['id']
            outputs = model(pixel_values=images)
            embeddings = F.normalize(outputs.pooler_output, p=2, dim=-1)
            all_image_embeddings.append(embeddings.cpu().numpy())
            all_image_ids.extend(image_ids)

    all_image_embeddings = np.vstack(all_image_embeddings)
    print(f"Processed {len(all_image_ids)} images.")
    print(f"Embedding matrix shape: {all_image_embeddings.shape}")

    embedding_dim = config['model']['embedding_dim']
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(all_image_embeddings)
    
    faiss.write_index(index, os.path.join(output_dir, "image_features.index"))
    with open(os.path.join(output_dir, "image_ids.txt"), "w") as f:
        for image_id in all_image_ids:
            f.write(f"{image_id}\n")
            
    print(f"Faiss index and ID list saved to '{output_dir}'")

if __name__ == '__main__':
    create_database()

