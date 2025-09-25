import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import open_clip

# --- 1. 설정 (Configuration) ---
# 이 섹션에서 모델, 데이터 경로, 학습 파라미터를 설정합니다.
class Config:
    # 데이터셋 경로 설정
    # GLDv2 학습 이미지가 저장된 최상위 디렉토리
    IMAGE_DIR = "/home/workspace/data/GLDv2/train/image" 
    # 사용자 지정 메타데이터 CSV 파일 경로
    CSV_PATH = "/home/workspace/data/GLDv2/train/train_custom.csv"

    # 모델 설정
    # OpenCLIP에서 사용할 사전 학습 모델. 
    # 사용 가능한 모델 목록은 open_clip.list_pretrained()로 확인할 수 있습니다. [4]
    MODEL_NAME = 'ViT-B-32'
    PRETRAINED_WEIGHTS = 'laion2b_s34b_b79k'

    # 학습 하이퍼파라미터
    EPOCHS = 5
    BATCH_SIZE = 16 # GPU 메모리에 따라 조정
    LEARNING_RATE = 1e-6 # 파인튜닝 시에는 낮은 학습률을 사용하는 것이 중요합니다. [5]
    WEIGHT_DECAY = 0.1
    
    # 시스템 설정
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4 # 데이터 로딩을 위한 병렬 프로세스 수

# --- 2. 데이터셋 클래스 정의 ---
# PyTorch의 Dataset 클래스를 상속받아 우리 데이터에 맞게 커스터마이징합니다. [6, 7]
class GLDv2CustomDataset(Dataset):
    def __init__(self, image_dir, csv_path, image_preprocess):
        """
        Args:
            image_dir (str): 이미지 파일이 있는 디렉토리 경로.
            csv_path (str): 이미지 ID와 설명을 담은 CSV 파일 경로.
            image_preprocess (callable): CLIP 모델에 맞는 이미지 전처리기.
        """
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.image_preprocess = image_preprocess

    def __len__(self):
        """데이터셋의 전체 샘플 수를 반환합니다."""
        return len(self.df)

    def __getitem__(self, idx):
        """주어진 인덱스(idx)에 해당하는 샘플(이미지, 텍스트)을 반환합니다."""
        # CSV 파일에서 이미지 ID와 설명을 가져옵니다.
        image_id = self.df.iloc[idx]['id']
        description = self.df.iloc[idx]['description']

        # GLDv2의 디렉토리 구조에 따라 이미지 경로를 구성합니다.
        # 예: '4e7c3c4e083ee269' -> '4/e/7/4e7c3c4e083ee269.jpg'
        image_path = os.path.join(self.image_dir, image_id[0], image_id[1], image_id[2], f"{image_id}.jpg")
        
        try:
            # 이미지 파일을 열고 RGB로 변환합니다.
            image = Image.open(image_path).convert("RGB")
            # CLIP 모델의 전처리기를 적용합니다.
            image_tensor = self.image_preprocess(image)
        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            # 파일이 없을 경우, 빈 텐서와 빈 설명을 반환하여 배치 생성 시 걸러낼 수 있도록 합니다.
            return None, None

        return image_tensor, description

# --- 3. 대조 손실 함수 (Contrastive Loss) ---
# CLIP의 핵심 학습 목표인 InfoNCE 손실을 구현합니다. [8, 9]
def contrastive_loss(image_features, text_features, logit_scale):
    """
    이미지와 텍스트 임베딩 간의 대조 손실을 계산합니다.
    Args:
        image_features (torch.Tensor): 이미지 임베딩 텐서.
        text_features (torch.Tensor): 텍스트 임베딩 텐서.
        logit_scale (torch.Tensor): 유사도 점수를 조절하는 학습 가능한 파라미터.
    """
    # 정규화된 임베딩 간의 코사인 유사도를 계산합니다.
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    # 배치 내에서 올바른 (이미지, 텍스트) 쌍을 찾기 위한 정답 레이블을 생성합니다.
    # (0, 1, 2,..., batch_size-1)
    batch_size = image_features.shape[0]
    ground_truth = torch.arange(batch_size, dtype=torch.long, device=Config.DEVICE)

    # 이미지 기준 손실과 텍스트 기준 손실을 각각 계산하고 평균을 냅니다.
    loss_img = F.cross_entropy(logits_per_image, ground_truth)
    loss_txt = F.cross_entropy(logits_per_text, ground_truth)
    total_loss = (loss_img + loss_txt) / 2
    
    return total_loss

# --- 4. 학습 루프 ---
def train_one_epoch(model, dataloader, optimizer, tokenizer, device):
    model.train()
    total_loss = 0.0
    
    # tqdm을 사용하여 진행 상황을 시각적으로 표시합니다.
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, texts in pbar:
        # 데이터 로딩 중 파일이 없는 경우를 처리합니다.
        if images is None:
            continue
            
        # 데이터를 지정된 장치(GPU/CPU)로 이동합니다.
        images = images.to(device)
        
        # 텍스트를 토큰화하고 장치로 이동합니다. [10]
        text_tokens = tokenizer(texts).to(device)

        # 모델의 순전파(forward pass)를 통해 이미지와 텍스트의 임베딩을 얻습니다.
        image_features, text_features, logit_scale = model(images, text_tokens)
        
        # 손실을 계산합니다.
        loss = contrastive_loss(image_features, text_features, logit_scale)
        
        # 역전파(backward pass) 및 가중치 업데이트 [11, 12]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        
    return total_loss / len(dataloader)

# --- 5. 메인 실행 함수 ---
def main():
    print(f"Using device: {Config.DEVICE}")

    # 1. 모델 및 전처리기 불러오기
    print("Loading OpenCLIP model...")
    # `open_clip.create_model_and_transforms`는 모델, 학습용 전처리기, 평가용 전처리기를 반환합니다.
    # 파인튜닝 시에는 학습용 전처리기(데이터 증강 포함)를 사용하는 것이 좋습니다.
    model, _, preprocess = open_clip.create_model_and_transforms(
        Config.MODEL_NAME, 
        pretrained=Config.PRETRAINED_WEIGHTS
    )
    model.to(Config.DEVICE)
    
    # 텍스트 토크나이저를 불러옵니다.
    tokenizer = open_clip.get_tokenizer(Config.MODEL_NAME)
    print("Model loaded.")

    # 2. 데이터셋 및 데이터로더 준비
    print("Preparing dataset...")
    # collate_fn을 정의하여 데이터 로딩 중 발생할 수 있는 None 값을 처리합니다.
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None, None
        images, texts = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, list(texts)

    dataset = GLDv2CustomDataset(
        image_dir=Config.IMAGE_DIR,
        csv_path=Config.CSV_PATH,
        image_preprocess=preprocess
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
    # AdamW 옵티마이저를 사용합니다. [13]
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )

    # 4. 학습 시작
    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        avg_loss = train_one_epoch(model, dataloader, optimizer, tokenizer, Config.DEVICE)
        print(f"Epoch {epoch+1}/{Config.EPOCHS}, Average Loss: {avg_loss:.4f}")

    # 5. 학습된 모델 저장
    save_path = f"./finetuned_gldv2_clip_{Config.MODEL_NAME}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model fine-tuning complete. Saved to {save_path}")


if __name__ == "__main__":
    main()
