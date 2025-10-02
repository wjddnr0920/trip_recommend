import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
# transformers 라이브러리에서 필요한 클래스들을 임포트합니다.
from transformers import CLIPProcessor, CLIPModel

# --- 1. 설정 (Configuration) ---
class Config:
    # 데이터셋 경로 설정
    IMAGE_DIR = "/home/workspace/data/GLDv2/train/image" 
    CSV_PATH = "/home/workspace/data/GLDv2/train/train_custom.csv"

    # 모델 설정 (Hugging Face 모델 ID)
    # open_clip의 'ViT-B-32'와 'laion2b...'에 해당하는
    # 가장 표준적인 CLIP 모델은 'openai/clip-vit-base-patch32' 입니다.
    MODEL_ID = 'openai/clip-vit-base-patch32'

    # 학습 하이퍼파라미터
    EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-6
    WEIGHT_DECAY = 0.1
    
    # 시스템 설정
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

# --- 2. 데이터셋 클래스 정의 ---
class GLDv2CustomDataset(Dataset):
    def __init__(self, image_dir, csv_path, processor):
        """
        Args:
            image_dir (str): 이미지 파일이 있는 디렉토리 경로.
            csv_path (str): 이미지 ID와 설명을 담은 CSV 파일 경로.
            processor (CLIPProcessor): 이미지 전처리 및 텍스트 토큰화를 위한 프로세서.
        """
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
            # processor를 사용하여 이미지를 전처리합니다.
            # return_tensors="pt"는 PyTorch 텐서를 반환하도록 합니다.
            # squeeze()는 불필요한 배치 차원을 제거합니다.
            image_tensor = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        except FileNotFoundError:
            # 파일을 찾지 못하면 None을 반환하여 collate_fn에서 처리하도록 합니다.
            return None, None

        return image_tensor, description

# --- 3. 대조 손실 함수 (Contrastive Loss) ---
# 이 함수는 라이브러리에 의존하지 않으므로 변경할 필요가 없습니다.
def contrastive_loss(image_features, text_features, logit_scale):
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    batch_size = image_features.shape[0]
    ground_truth = torch.arange(batch_size, dtype=torch.long, device=Config.DEVICE)

    loss_img = F.cross_entropy(logits_per_image, ground_truth)
    loss_txt = F.cross_entropy(logits_per_text, ground_truth)
    total_loss = (loss_img + loss_txt) / 2
    
    return total_loss

# --- 4. 학습 루프 ---
def train_one_epoch(model, dataloader, optimizer, processor, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, texts in pbar:
        # collate_fn에서 걸러진 배치를 건너뜁니다.
        if images is None:
            continue
            
        images = images.to(device)
        
        # processor를 사용하여 텍스트를 토큰화합니다.
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)

        # 모델의 순전파(forward pass)
        # transformers 모델은 각 특징을 별도로 추출해야 합니다.
        image_features = model.get_image_features(pixel_values=images)
        text_features = model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        
        # 정규화
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # logit_scale은 모델의 학습 가능한 파라미터입니다.
        logit_scale = model.logit_scale.exp()
        
        loss = contrastive_loss(image_features, text_features, logit_scale)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        
    return total_loss / len(dataloader)

# --- 5. 메인 실행 함수 ---
def main():
    print(f"Using device: {Config.DEVICE}")

    # 1. 모델 및 프로세서 불러오기
    print("Loading CLIP model and processor from Hugging Face...")
    # CLIPProcessor는 이미지 전처리기와 텍스트 토크나이저를 모두 포함합니다.
    processor = CLIPProcessor.from_pretrained(Config.MODEL_ID, use_fast=True)
    model = CLIPModel.from_pretrained(Config.MODEL_ID).to(Config.DEVICE)
    print("Model loaded.")

    # 2. 데이터셋 및 데이터로더 준비
    print("Preparing dataset...")
    # 데이터 로딩 중 발생할 수 있는 None 값을 안전하게 처리하는 collate_fn
    def collate_fn(batch):
        # (이미지, 텍스트) 쌍에서 이미지가 None이 아닌 것만 필터링
        batch = list(filter(lambda x: x[0] is not None, batch))
        if not batch:
            return None, None
        images, texts = zip(*batch)
        # 이미지들을 하나의 텐서로 합침
        images = torch.stack(images, dim=0)
        return images, list(texts)

    dataset = GLDv2CustomDataset(
        image_dir=Config.IMAGE_DIR,
        csv_path=Config.CSV_PATH,
        processor=processor
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    print(f"Dataset prepared with {len(dataset)} samples.")

    # 3. 옵티마이저 설정
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )

    # 4. 학습 시작
    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # train_one_epoch 함수에 tokenizer 대신 processor를 전달합니다.
        avg_loss = train_one_epoch(model, dataloader, optimizer, processor, Config.DEVICE)
        print(f"Epoch {epoch+1}/{Config.EPOCHS}, Average Loss: {avg_loss:.4f}")

    # 5. 학습된 모델 저장
    save_path = f"./finetuned_gldv2_clip_{Config.MODEL_ID.replace('/', '_')}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model fine-tuning complete. Saved to {save_path}")

if __name__ == "__main__":
    main()
