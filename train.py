import os
import yaml # YAML 라이브러리 임포트
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import random
from torch.amp import GradScaler, autocast

# --- 경고 해결을 위한 코드 추가 ---
# DataLoader의 멀티 프로세싱과 충돌을 방지하기 위해 
# 토크나이저의 병렬 처리를 비활성화합니다.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 추가된 함수: 시드 고정 ---
def set_seed(seed):
    """실험 재현성을 위해 랜덤 시드를 고정하는 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # CUDNN 설정 (재현성에 영향을 줄 수 있음)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Config Class Removed ---

class GLDv2CustomDataset(Dataset):
    def __init__(self, image_dir, csv_path, processor):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]['id']
        description = self.df.iloc[idx]['description']
        image_path = os.path.join(self.image_dir, image_id[0], image_id[1], image_id[2], f"{image_id}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        except FileNotFoundError:
            return None, None
        return image_tensor, description

def contrastive_loss(image_features, text_features, logit_scale):
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T
    batch_size = image_features.shape[0]
    ground_truth = torch.arange(batch_size, dtype=torch.long, device=image_features.device)
    loss_img = F.cross_entropy(logits_per_image, ground_truth)
    loss_txt = F.cross_entropy(logits_per_text, ground_truth)
    return (loss_img + loss_txt) / 2

# --- 수정된 함수: 학습 루프에 AMP 로직 추가 ---
def train_one_epoch(model, dataloader, optimizer, processor, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, texts in pbar:
        if images is None:
            continue
        images = images.to(device, non_blocking=True)
        inputs = processor.tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True)
        # 토큰화된 결과도 non_blocking으로 이동
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        # autocast 컨텍스트 매니저: 이 블록 내의 연산을 자동으로 혼합 정밀도로 수행
        with autocast(enabled=use_amp, device_type=device):
            image_features = F.normalize(model.get_image_features(pixel_values=images), p=2, dim=-1)
            text_features = F.normalize(model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask), p=2, dim=-1)
            logit_scale = model.logit_scale.exp()
            loss = contrastive_loss(image_features, text_features, logit_scale)

        optimizer.zero_grad()
        
        # GradScaler를 사용하여 스케일링된 손실로 역전파
        scaler.scale(loss).backward()
        # 스케일링된 그래디언트로 옵티마이저 업데이트
        scaler.step(optimizer)
        # 다음 반복을 위해 스케일러 업데이트
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    return total_loss / len(dataloader)

def main():
    with open('/home/workspace/config_custom.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- 추가된 부분: 시드 고정 함수 호출 ---
    set_seed(config['system']['seed'])
    print(f"Random seed set to {config['system']['seed']}")

    device = "cuda" if torch.cuda.is_available() and config['system']['device'] == 'auto' else "cpu"
    print(f"Using device: {device}")

    # --- 추가된 부분: AMP 설정 확인 ---
    use_amp = config['training']['use_amp']
    print(f"Automatic Mixed Precision (AMP) is {'ENABLED' if use_amp else 'DISABLED'}")

    print("Loading CLIP model and processor...")
    processor = CLIPProcessor.from_pretrained(config['model']['model_id'], use_fast=True)
    model = CLIPModel.from_pretrained(config['model']['model_id']).to(device)
    print("Model loaded.")

    def collate_fn(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        if not batch: return None, None
        images, texts = zip(*batch)
        return torch.stack(images, dim=0), list(texts)

    dataset = GLDv2CustomDataset(
        image_dir=config['paths']['train_image_dir'],
        csv_path=config['paths']['train_csv_path'],
        processor=processor
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['system']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    print(f"Dataset prepared with {len(dataset)} samples.")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    
    # --- 추가된 부분: GradScaler 초기화 ---
    # enabled=False로 설정하면 scaler의 모든 연산이 비활성화(no-op)됩니다.
    scaler = GradScaler(enabled=use_amp)

    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        # --- 수정된 부분: 학습 함수에 scaler와 use_amp 전달 ---
        avg_loss = train_one_epoch(model, dataloader, optimizer, processor, scaler, device, use_amp)
        print(f"Epoch {epoch+1}/{config['training']['epochs']}, Average Loss: {avg_loss:.4f}")
    
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    model_name_safe = config['model']['model_id'].replace('/', '_')
    save_path = os.path.join(config['paths']['output_dir'], f"finetuned_{model_name_safe}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model fine-tuning complete. Saved to {save_path}")

if __name__ == "__main__":
    main()
