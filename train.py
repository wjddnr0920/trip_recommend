import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import random
from torch.amp import GradScaler, autocast

# --- DALI 라이브러리 임포트 ---
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

def set_seed(seed):
    """실험 재현성을 위해 랜덤 시드를 고정하는 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- DALI 학습 파이프라인 정의 ---
class DALITrainPipeline(Pipeline):
    def __init__(self, image_paths, batch_size, num_threads, device_id, processor):
        super(DALITrainPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        
        self.image_paths = image_paths
        
        # Processor로부터 모든 전처리 값을 동적으로 설정
        image_proc = processor.image_processor
        self.mean = image_proc.image_mean
        self.std = image_proc.image_std
        self.crop_size = image_proc.crop_size['height']
        
    def define_graph(self):
        # 파일 리더가 이미지와 함께 원본 인덱스를 '레이블'로 반환
        jpegs, labels = fn.readers.file(
            files=self.image_paths, 
            name="file_reader", 
            random_shuffle=True # 학습 시에는 데이터를 섞어주는 것이 좋음
        )
        
        # --- 데이터 증강 (Data Augmentation) 적용 ---
        # 1. RandomResizedCrop과 동일한 효과
        # 이미지를 디코딩하면서 동시에 랜덤한 영역을 잘라냄
        images = fn.decoders.image_random_crop(
            jpegs,
            device="mixed",
            output_type=types.RGB
        )
        # 2. 리사이즈
        images = fn.resize(images, size=self.crop_size)
        # 3. RandomHorizontalFlip (50% 확률)
        do_flip = fn.random.coin_flip(probability=0.5)
        images = fn.flip(images, horizontal=do_flip)

        output_dtype = types.FLOAT16 if use_amp else types.FLOAT
        # 4. 정규화
        images = fn.crop_mirror_normalize(
            images,
            dtype=output_dtype,
            mean=self.mean,
            std=self.std,
            output_layout="CHW"
        )
        return images, labels

def contrastive_loss(image_features, text_features, logit_scale):
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T
    batch_size = image_features.shape[0]
    ground_truth = torch.arange(batch_size, dtype=torch.long, device=image_features.device)
    loss_img = F.cross_entropy(logits_per_image, ground_truth)
    loss_txt = F.cross_entropy(logits_per_text, ground_truth)
    return (loss_img + loss_txt) / 2

def train_one_epoch(model, dataloader, optimizer, processor, scaler, device, use_amp, valid_texts):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training with DALI", leave=False)
    for batch in pbar:
        # DALI 이터레이터는 [이미지 딕셔너리, 레이블 딕셔너리]를 반환
        images = batch[0]['data']
        labels = batch[0]['label'].squeeze(-1).long().tolist() # 레이블(인덱스) 추출
        
        # DALI가 반환한 인덱스를 사용하여 현재 배치의 텍스트를 가져옴
        texts = [valid_texts[i] for i in labels]
        
        # 이미지는 이미 GPU에 있으므로 .to(device) 불필요
        tokenized_inputs = processor.tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device, non_blocking=True) for k, v in tokenized_inputs.items()}

        with autocast(enabled=use_amp, device_type='cuda'):
            image_features = F.normalize(model.get_image_features(pixel_values=images), p=2, dim=-1)
            text_features = F.normalize(model.get_text_features(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask']
            ), p=2, dim=-1)
            logit_scale = model.logit_scale.exp()
            loss = contrastive_loss(image_features, text_features, logit_scale)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        
    return total_loss / len(dataloader)

def main():
    with open('/home/workspace/config_custom.yaml', 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['system']['seed'])
    print(f"Random seed set to {config['system']['seed']}")

    device_id = 0
    device = f"cuda:{device_id}" if torch.cuda.is_available() and config['system']['device'] == 'auto' else "cpu"
    if device == "cpu":
        print("Error: NVIDIA DALI requires a GPU to run.")
        return
        
    global use_amp
    use_amp = config['training']['use_amp'] and device == 'cuda'
    print(f"Using device: {device}, AMP: {'ENABLED' if use_amp else 'DISABLED'}")

    print("Loading CLIP model and processor...")
    processor = CLIPProcessor.from_pretrained(config['model']['model_id'], use_fast=True)
    model = CLIPModel.from_pretrained(config['model']['model_id']).to(device)
    print("Model loaded.")

    # --- 데이터 로딩 방식 변경: PyTorch -> DALI ---
    print("Preparing DALI pipeline for training...")
    df = pd.read_csv(config['paths']['train_csv_path'])
    
    # 유효한 이미지 경로와 텍스트 리스트를 미리 생성
    valid_paths, valid_texts = [], []
    print("Filtering valid training image paths...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(config['paths']['train_image_dir'], row['id'][0], row['id'][1], row['id'][2], f"{row['id']}.jpg")
        if os.path.exists(img_path):
            valid_paths.append(img_path)
            valid_texts.append(row['description'])
    print(f"Found {len(valid_paths)} valid training images.")

    pipeline = DALITrainPipeline(
        image_paths=valid_paths, 
        batch_size=config['training']['batch_size'],
        num_threads=config['system']['num_workers'], 
        device_id=device_id,
        processor=processor
    )
    pipeline.build()

    dali_iterator = DALIClassificationIterator(
        pipelines=[pipeline], reader_name="file_reader",
        last_batch_policy=LastBatchPolicy.DROP, # 마지막 배치가 작으면 버림 (학습 안정성)
        auto_reset=True
    )
    
    print(f"DALI pipeline ready with {len(dali_iterator)} batches per epoch.")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    scaler = GradScaler(enabled=use_amp)

    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        avg_loss = train_one_epoch(model, dali_iterator, optimizer, processor, scaler, device, use_amp, valid_texts)
        print(f"Epoch {epoch+1}/{config['training']['epochs']}, Average Loss: {avg_loss:.4f}")
    
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    model_name_safe = config['model']['model_id'].replace('/', '_')
    save_path = os.path.join(output_dir, f"finetuned_{model_name_safe}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model fine-tuning complete. Saved to {save_path}")

if __name__ == "__main__":
    main()

