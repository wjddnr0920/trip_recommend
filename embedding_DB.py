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
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color='black')
            processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            image_id = "file_not_found"
        return {"image": processed_image, "id": image_id}

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
    dataloader = DataLoader(dataset, batch_size=config['retrieval']['batch_size'], shuffle=False)

    all_image_embeddings = []
    all_image_ids = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Creating index image embeddings"):
            images = batch['image'].to(device)
            image_ids = batch['id']
            
            with autocast(enabled=use_amp, device_type=device):
                embeddings = model.get_image_features(pixel_values=images)
            
            all_image_embeddings.append(embeddings.cpu().float().numpy())
            all_image_ids.extend(image_ids)

    all_image_embeddings = np.vstack(all_image_embeddings)
    print(f"Processed {len(all_image_ids)} images.")
    print(f"Embedding matrix shape: {all_image_embeddings.shape}")

    # --- 수정된 부분 ---
    # 모델의 config에서 직접 임베딩 차원을 가져옵니다.
    embedding_dim = model.config.projection_dim
    print(f"Embedding dimension detected from model config: {embedding_dim}")

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(all_image_embeddings)
    
    faiss.write_index(index, os.path.join(output_dir, "image_features.index"))
    with open(os.path.join(output_dir, "image_ids.txt"), "w") as f:
        for image_id in all_image_ids:
            f.write(f"{image_id}\n")
            
    print(f"Faiss index and ID list saved to '{output_dir}'")

if __name__ == '__main__':
    create_database()

