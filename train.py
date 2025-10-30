import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
import numpy as np
import random
from torch.amp import GradScaler, autocast
import shutil

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

def set_seed(seed):
    """실험 재현성을 위해 랜덤 시드를 고정하는 함수 (DALI 외 부분)"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DALITrainPipeline(Pipeline):
    """학습을 위한 DALI 파이프라인 (데이터 증강 포함)"""
    # --- 수정: __init__에 epoch_seed 추가 ---
    def __init__(self, image_paths, batch_size, num_threads, device_id, processor, epoch_seed):
        # --- 수정: super().__init__에 epoch_seed 전달 ---
        super(DALITrainPipeline, self).__init__(batch_size, num_threads, device_id, seed=epoch_seed)
        self.image_paths = image_paths
        image_proc = processor.image_processor
        self.mean = image_proc.image_mean
        self.std = image_proc.image_std
        if hasattr(image_proc, 'crop_size') and isinstance(image_proc.crop_size, dict):
            self.crop_size = image_proc.crop_size['height']
        else:
            self.crop_size = image_proc.size['height']
        
    def define_graph(self):
        # --- 수정: shard_id와 num_shards 추가 (분산 학습 대비 및 안정성) ---
        # 외부에서 shard 정보를 받지 않으므로 0, 1로 고정 (단일 GPU 환경)
        jpegs, labels = fn.readers.file(
            files=self.image_paths, 
            name="file_reader", 
            random_shuffle=True, 
            shard_id=0, 
            num_shards=1 
        )
        images = fn.decoders.image_random_crop(jpegs, device="mixed", output_type=types.RGB)
        images = fn.resize(images, size=self.crop_size)
        do_flip = fn.random.coin_flip(probability=0.5)
        images = fn.flip(images, horizontal=do_flip)
        output_dtype = types.FLOAT
        images = fn.crop_mirror_normalize(
            images, dtype=output_dtype,
            mean=[m * 255.0 for m in self.mean],
            std=[s * 255.0 for s in self.std],
            output_layout="CHW"
        )
        return images, labels

def train_one_epoch(model, dataloader, optimizer, processor, scaler, device, use_amp, valid_texts, epoch, total_epochs, model_id, start_epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} Training", leave=True)
    
    model_id_lower = model_id.lower()
    padding_strategy = True; tokenizer_kwargs_extra = {}; print_msg = "CLIP padding"
    if "google/siglip2" in model_id_lower:
        padding_strategy = "max_length"; tokenizer_kwargs_extra["max_length"] = 64; print_msg = "SigLIP2 padding (max_64)"
    elif "google/siglip" in model_id_lower:
        padding_strategy = "max_length"; print_msg = "SigLIP padding (max_length)"
            
    for batch_idx, batch in enumerate(pbar):
        # 첫 배치 시작 시 패딩 전략 한 번만 출력
        if batch_idx == 0 and epoch == (start_epoch + 1):
             print(f"Tokenizer padding strategy: {print_msg}")

        images, labels = batch[0]['data'], batch[0]['label'].squeeze(-1).long().tolist()
        texts = [valid_texts[i] for i in labels]
        
        tokenized_inputs = processor.tokenizer(
            text=texts, return_tensors="pt", padding=padding_strategy,
            truncation=True, **tokenizer_kwargs_extra
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in tokenized_inputs.items()}

        with autocast(enabled=use_amp, device_type='cuda'):
            outputs = model(**inputs, pixel_values=images, return_loss=True)
            loss = outputs.loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def save_checkpoint(epoch, model, optimizer, scaler, config):
    output_dir = config['paths']['output_dir']
    checkpoint_state = {
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict()
    }
    filename = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
    os.makedirs(os.path.dirname(filename), exist_ok=True) 
    torch.save(checkpoint_state, filename)
    last_filename = os.path.join(output_dir, f"checkpoint_last.pt")
    shutil.copyfile(filename, last_filename)
    # print(f"Checkpoint saved to {filename} and {last_filename}") # 로그 줄이기

def main():
    with open('/home/workspace/config_custom.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- 수정: 기본 시드 설정 ---
    base_seed = config['system']['seed']
    set_seed(base_seed) # DALI 외 PyTorch, Numpy 등 시드 고정
    print(f"Base random seed set to {base_seed}")

    device_id = 0
    device = f"cuda:{device_id}" if torch.cuda.is_available() and config['system']['device'] == 'auto' else "cpu"
    if device == "cpu": print("Error: DALI requires GPU."); return
        
    use_amp = config['training']['use_amp'] and device.startswith('cuda')
    print(f"Using device: {device}, AMP: {'ENABLED' if use_amp else 'DISABLED'}")

    model_id = config['model']['model_id']
    print(f"Loading model and processor for: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id).to(device)
    print("Model loaded.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    scaler = GradScaler(enabled=use_amp)
    
    global start_epoch
    start_epoch = 0
    resume_checkpoint_path = config['training'].get('resume_from_checkpoint')

    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict']) 
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] 
        print(f"Resumed from epoch {start_epoch}. Starting next epoch.")
    else:
        if resume_checkpoint_path: print(f"Warning: Checkpoint file not found. Starting from scratch.")
        else: print("No checkpoint specified. Starting from scratch.")

    print("Preparing DALI pipeline assets...")
    df = pd.read_csv(config['paths']['train_csv_path'])
    valid_paths, valid_texts = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering train paths"):
        img_path = os.path.join(config['paths']['train_image_dir'], str(row['id'])[0], str(row['id'])[1], str(row['id'])[2], f"{row['id']}.jpg")
        if os.path.exists(img_path):
            valid_paths.append(img_path); valid_texts.append(row['description'])
    print(f"Found {len(valid_paths)} valid training images.")

    print("\nStarting training...")
    total_epochs = config['training']['epochs']
    save_interval = config['training'].get('save_checkpoint_interval', 1)

    # --- 수정: 에포크 루프 내에서 파이프라인 및 이터레이터 생성 ---
    for epoch in range(start_epoch, total_epochs):
        current_epoch = epoch + 1
        
        # 에포크별 시드 계산
        epoch_seed = base_seed + current_epoch - 1
        print(f"\n--- Starting Epoch {current_epoch}/{total_epochs} with DALI seed: {epoch_seed} ---")

        # 현재 에포크 시드로 파이프라인 생성
        pipeline = DALITrainPipeline(
            image_paths=valid_paths, batch_size=config['training']['batch_size'],
            num_threads=config['system']['num_workers'], device_id=device_id,
            processor=processor,
            epoch_seed=epoch_seed # 에포크 시드 전달
        )
        pipeline.build()

        # 새로 생성된 파이프라인으로 이터레이터 생성
        dali_iterator = DALIClassificationIterator(
            pipelines=[pipeline], reader_name="file_reader",
            last_batch_policy=LastBatchPolicy.DROP, auto_reset=True
        )
        
        avg_loss = train_one_epoch(
            model, dali_iterator, optimizer, processor, scaler, 
            device, use_amp, valid_texts, current_epoch, total_epochs, model_id, start_epoch
        )
        print(f"--- Epoch {current_epoch}/{total_epochs} Complete --- Average Loss: {avg_loss:.4f}\n")
        
        # 체크포인트 저장
        if current_epoch % save_interval == 0:
            save_checkpoint(current_epoch, model, optimizer, scaler, config)
            
        # --- 추가: 메모리 관리를 위해 사용 완료된 이터레이터와 파이프라인 삭제 ---
        del dali_iterator
        del pipeline
            
    # 최종 모델 가중치만 저장
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True) 
    model_name_safe = model_id.replace('/', '_')
    final_save_path = os.path.join(output_dir, f"finetuned_{model_name_safe}_final.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model state dict saved to {final_save_path}")

    print("Model fine-tuning complete.")

if __name__ == "__main__":
    main()

