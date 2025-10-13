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
    
    # --- 추가된 부분: 추론용 AMP 설정 확인 ---
    use_amp = config['retrieval']['use_amp']
    print(f"Inference with AMP is {'ENABLED' if use_amp else 'DISABLED'}")

    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model and processor...")
    processor = CLIPProcessor.from_pretrained(config['model']['model_id'], use_fast=True)
    
    finetuned_path = config['model']['finetuned_path']
    if finetuned_path and os.path.exists(finetuned_path):
        print(f"Loading fine-tuned model from: {finetuned_path}")
        model_to_load = CLIPModel.from_pretrained(config['model']['model_id'])
        model_to_load.load_state_dict(torch.load(finetuned_path))
        model = model_to_load.vision_model.to(device)
    else:
        print("Fine-tuned model not found. Using pre-trained weights.")
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
            
            # --- 수정된 부분: autocast 컨텍스트 적용 ---
            # 이 블록 안의 연산이 자동으로 float16으로 수행됩니다.
            with autocast(enabled=use_amp, device_type=device):
                outputs = model(pixel_values=images)
                embeddings = F.normalize(outputs.pooler_output, p=2, dim=-1)
            
            # CPU로 옮기기 전에 float32로 다시 변환해주는 것이 안정적일 수 있습니다.
            all_image_embeddings.append(embeddings.cpu().float().numpy())
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
