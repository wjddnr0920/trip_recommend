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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 최종 완성된 DALI 파이프라인 ---
class DALITrainPipeline(Pipeline):
    """
    학습을 위한 DALI 파이프라인.
    AutoProcessor로부터 전처리 값을 동적으로 읽어와 모든 모델에 자동 대응합니다.
    """
    def __init__(self, image_paths, batch_size, num_threads, device_id, processor):
        super(DALITrainPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.image_paths = image_paths
        image_proc = processor.image_processor
        
        # --- 핵심 개선: if/elif 제거! ---
        # processor가 CLIP이든 SigLIP이든, 알맞은 mean/std 값을 제공합니다.
        self.mean = image_proc.image_mean
        self.std = image_proc.image_std
        
        # 1. crop_size가 유효한 딕셔너리인지 먼저 확인합니다.
        if hasattr(image_proc, 'crop_size') and isinstance(image_proc.crop_size, dict):
            self.crop_size = image_proc.crop_size['height']
        # 2. 그렇지 않으면(None이거나 없는 경우), size 속성을 사용합니다.
        else:
            self.crop_size = image_proc.size['height']
        
    def define_graph(self):
        jpegs, labels = fn.readers.file(files=self.image_paths, name="file_reader", random_shuffle=True)
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

def train_one_epoch(model, dataloader, optimizer, processor, scaler, device, use_amp, valid_texts, epoch, total_epochs, model_id):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} Training", leave=True)

    model_id_lower = model_id.lower()
    
    for batch in pbar:
        images, labels = batch[0]['data'], batch[0]['label'].squeeze(-1).long().tolist()
        texts = [valid_texts[i] for i in labels]

        tokenizer_kwargs = {
            "text": texts,
            "return_tensors": "pt",
            "truncation": True 
        }
        
        if "google/siglip2" in model_id_lower:
            # SigLIP2: max_length=64 고정
            tokenizer_kwargs["padding"] = "max_length"
            tokenizer_kwargs["max_length"] = 64 
            print_msg = "Using SigLIP2 padding (max_length=64)"
        elif "google/siglip" in model_id_lower:
            # 일반 SigLIP: max_length 고정 (Processor 기본값 따름)
            tokenizer_kwargs["padding"] = "max_length"
            print_msg = "Using SigLIP padding (max_length)"
        else: # 기본값 또는 CLIP
            tokenizer_kwargs["padding"] = True # 배치별 동적 패딩
            print_msg = "Using CLIP padding (dynamic)"

        # (디버깅용) 첫 배치에서만 패딩 전략 출력
        if pbar.n == 0:
             print(f"Tokenizer padding strategy: {print_msg}")
            
        tokenized_inputs = processor.tokenizer(**tokenizer_kwargs)
        
        inputs = {k: v.to(device, non_blocking=True) for k, v in tokenized_inputs.items()}

        with autocast(enabled=use_amp, device_type='cuda'):
            # 1. 텍스트 입력을 위한 딕셔너리 준비 (inputs 자체를 사용)
            text_inputs = inputs
            
            # 2. **를 사용하여 모델에 인자 전달
            # CLIP이면 {'input_ids': ..., 'attention_mask': ...}가 전달됨
            # SigLIP이면 {'input_ids': ...}만 전달됨
            outputs = model(
                **text_inputs,
                pixel_values=images,
                return_loss=True
            )
            loss = outputs.loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def main():
    with open('/home/workspace/config_custom.yaml', 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['system']['seed'])
    device_id = 0
    device = f"cuda:{device_id}" if torch.cuda.is_available() and config['system']['device'] == 'auto' else "cpu"
    if device == "cpu":
        print("Error: NVIDIA DALI requires a GPU to run."); return
        
    use_amp = config['training']['use_amp'] and device.startswith('cuda')
    print(f"Using device: {device}, AMP: {'ENABLED' if use_amp else 'DISABLED'}")

    model_id = config['model']['model_id']
    print(f"Loading model and processor for: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id).to(device)
    print("Model loaded.")

    print("Preparing DALI pipeline for training...")
    df = pd.read_csv(config['paths']['train_csv_path'])
    valid_paths, valid_texts = [], []
    print("Filtering valid training image paths...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(config['paths']['train_image_dir'], str(row['id'])[0], str(row['id'])[1], str(row['id'])[2], f"{row['id']}.jpg")
        if os.path.exists(img_path):
            valid_paths.append(img_path)
            valid_texts.append(row['description'])
    print(f"Found {len(valid_paths)} valid training images.")

    pipeline = DALITrainPipeline(
        image_paths=valid_paths, batch_size=config['training']['batch_size'],
        num_threads=config['system']['num_workers'], device_id=device_id,
        processor=processor
    )
    pipeline.build()

    dali_iterator = DALIClassificationIterator(
        pipelines=[pipeline], reader_name="file_reader",
        last_batch_policy=LastBatchPolicy.DROP, auto_reset=True
    )
    print(f"DALI pipeline ready with {len(dali_iterator)} batches per epoch.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    scaler = GradScaler(enabled=use_amp)

    print("\nStarting training...")
    total_epochs = config['training']['epochs']
    for epoch in range(total_epochs):
        current_epoch = epoch + 1
        avg_loss = train_one_epoch(model, dali_iterator, optimizer, processor, scaler, device, use_amp, valid_texts, current_epoch, total_epochs, model_id)
        print(f"--- Epoch {current_epoch}/{total_epochs} Complete --- Average Loss: {avg_loss:.4f}\n")
    
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    model_name_safe = model_id.replace('/', '_')
    save_path = os.path.join(output_dir, f"finetuned_{model_name_safe}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model fine-tuning complete. Saved to {save_path}")

if __name__ == "__main__":
    main()
