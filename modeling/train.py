import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 토크나이저 병렬 처리 비활성화(DALI 충돌 방지)

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
    """랜덤 시드를 고정(DALI 제외)"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DALITrainPipeline(Pipeline):
    """
    학습을 위한 DALI 파이프라인 (데이터 증강 포함)
    image_paths: 이미지 파일 경로 리스트
    batch_size: 배치 크기
    num_threads: CPU 스레드 개수
    device_id: GPU 장치 ID
    processor: HuggingFace AutoProcessor 객체
    epoch_seed: 에포크별 시드 (추가 학습 시 랜덤성 유지)
    """
    def __init__(self, image_paths, batch_size, num_threads, device_id, processor, epoch_seed):
        super(DALITrainPipeline, self).__init__(batch_size, num_threads, device_id, seed=epoch_seed)
        self.image_paths = image_paths
        # processor에서 평균, 표준편차, 크기 추출(모델마다 다를 수 있음)
        image_proc = processor.image_processor
        self.mean = image_proc.image_mean
        self.std = image_proc.image_std
        # 모델(CLIP, SigLIP)마다 크기 속성명이 다를 수 있어 조건문 처리
        if hasattr(image_proc, 'crop_size') and isinstance(image_proc.crop_size, dict):
            self.crop_size = image_proc.crop_size['height']
        else:
            self.crop_size = image_proc.size['height']
        
    def define_graph(self):
        '''
        fn.readers.file: 이미지를 읽는 DALI 함수
        files: 이미지 파일 경로 리스트
        name: 이름표 지정 --> DALIClassificationIterator에서 참조(데이터의 개수 파악)
        random_shuffle: 데이터 셔플 여부
        shard_id, num_shards: 분산 학습 시 shard 설정 (여기서는 단일 GPU이므로 0, 1로 설정)
        output
        jpegs: 이미지 바이트(압축되어있음)
        labels: 정수 레이블 --> 이미지에 맞는 텍스트 설명(valid_texts)를 찾아내는 인덱스로 사용(train_one_epoch 함수에 활용)
        '''
        jpegs, labels = fn.readers.file(
            files=self.image_paths, 
            name="file_reader", 
            random_shuffle=True, 
            shard_id=0, 
            num_shards=1 
        )
        # 이미지를 읽음과 동시에 random crop 적용
        images = fn.decoders.image_random_crop(jpegs, device="mixed", output_type=types.RGB)
        # crop 후 모델 입력에 맞게 리사이즈
        images = fn.resize(images, size=self.crop_size)
        # 좌우 flip 증강 적용
        do_flip = fn.random.coin_flip(probability=0.5)
        images = fn.flip(images, horizontal=do_flip)
        output_dtype = types.FLOAT
        # 이미지 정규화 후 DALI의 범위(0~255)로 변환
        images = fn.crop_mirror_normalize(
            images, dtype=output_dtype,
            mean=[m * 255.0 for m in self.mean],
            std=[s * 255.0 for s in self.std],
            output_layout="CHW"
        )
        return images, labels

def train_one_epoch(model, dataloader, optimizer, processor, scaler, device, use_amp, valid_texts, epoch, total_epochs, model_id, start_epoch):
    '''
    한 에포크 동안 모델을 학습시키는 함수
    model: 학습할 모델
    dataloader: 여기선 DALIClassificationIterator 객체
    optimizer: 최적화 알고리즘
    processor: HuggingFace AutoProcessor 객체
    scaler: AMP용 GradScaler 객체
    device: 학습에 사용할 장치 (예: 'cuda:0')
    use_amp: AMP 사용 여부 (True/False)
    valid_texts: 이미지 레이블에 해당하는 텍스트 설명 리스트
    epoch: 현재 에포크 번호
    total_epochs: 총 에포크 수
    model_id: 모델 식별자 (예: 'openai/clip-vit-base-patch32')
    start_epoch: 학습 시작 에포크 번호 (체크포인트로 이어서 학습 시 활용)
    return: 해당 에포크의 평균 손실 값
    '''
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} Training", leave=True)
    
    model_id_lower = model_id.lower()
    # 모델마다 토크나이저의 패딩 설정이 달라짐
    tokenizer_kwargs_extra = {}
    # defualt는 CLIP 패딩
    print_msg = "CLIP padding"
    padding_strategy = True; 
    
    if "google/siglip2" in model_id_lower:
        padding_strategy = "max_length"; tokenizer_kwargs_extra["max_length"] = 64; print_msg = "SigLIP2 padding (max_64)"
    elif "google/siglip" in model_id_lower:
        padding_strategy = "max_length"; print_msg = "SigLIP padding (max_length)"
            
    for batch_idx, batch in enumerate(pbar):
        # 첫 배치 시작 시 디버깅을 위해 패딩 설정 한 번만 출력
        if batch_idx == 0 and epoch == (start_epoch + 1):
             print(f"Tokenizer padding strategy: {print_msg}")

        '''
        DALIClassificationIterator가 반환하는 batch의 구조
        [
            # 첫 번째 파이프라인(GPU 0)의 결과 딕셔너리
            {
                'data': <DALI Tensor (이미지 뭉치)>,
                'label': <DALI Tensor (번호표 뭉치)>
            }
        ]
        단일 GPU이므로 batch[0]만 사용

        batch[0]['label']: [[label1], [label2], ..., [labelN]] 형태
        squeeze(-1)로 [label1, label2, ..., labelN] 형태로 바꾼 후 정수형으로 변환 후 파이썬 리스트로 변환
        '''
        images = batch[0]['data']  # 이미지
        labels = batch[0]['label'].squeeze(-1).long().tolist()  # 몇 번째 이미지에 대한 인덱스 리스트
        texts = [valid_texts[i] for i in labels]  # 이미지에 해당하는 텍스트
        
        '''
        CLIP일 경우 padding=True
        SigLIP일 경우 padding='max_length'
        SigLIP2일 경우 padding='max_length', max_length=64
        huggingface에서 확인 가능
        '''
        tokenized_inputs = processor.tokenizer(
            text=texts, return_tensors="pt", padding=padding_strategy,
            truncation=True, **tokenizer_kwargs_extra
        )
        '''
        tokenized_inputs: {
            'input_ids': tensor([[101, 234, ...]]),      # 텍스트를 숫자로 바꾼 것 (CPU 텐서)
            'attention_mask': tensor([[1, 1, ...]])      # 패딩 여부를 표시한 것 (CPU 텐서)
        }
        .items(): 딕셔너리에서 (키, 값) 쌍을 반환
        첫 번째 루프 : k='input_ids', v=tensor([[101, 234, ...]])
        두 번째 루프 : k='attention_mask', v=tensor([[1, 1, ...]]) 
        v.to(device, non_blocking=True): 값을 GPU로 비동기 이동
        {k: ... for ...}: 이동이 완료된 텐서들을 원래 키 k와 짝지어 새로운 딕셔너리 생성
        '''
        inputs = {k: v.to(device, non_blocking=True) for k, v in tokenized_inputs.items()}

        with autocast(enabled=use_amp, device_type='cuda'):
            outputs = model(**inputs, pixel_values=images, return_loss=True)
            loss = outputs.loss

        optimizer.zero_grad()  # gradient 초기화
        scaler.scale(loss).backward()  # AMP를 사용하면 소수점 아래 작은 값은 무시되는 문제를 방지하기 위해 scale 적용 후 역전파
        scaler.step(optimizer)  # unscale하여 gradient 검사 후 optimizer.step() 대신 호출해서 가중치 업데이트(문제 발생 시 건너뜀)
        scaler.update()  # scaler 업데이트(학습 성공 유무에 따라 scale 값 조정)

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def save_checkpoint(epoch, model, optimizer, scaler, config):
    '''
    모델 학습 상태를 체크포인트로 저장하는 함수
    epoch: 현재 에포크 번호
    model: 학습 중인 모델
    optimizer: 최적화 알고리즘
    scaler: AMP용 GradScaler 객체
    config: config.yaml 파일
    '''
    output_dir = config['paths']['output_dir']
    # 가중치와 epoch, optimizer, scaler 상태를 모두 저장해 이어서 학습할 수 있도록 함
    checkpoint_state = {
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict()
    }
    filename = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
    os.makedirs(os.path.dirname(filename), exist_ok=True) 
    torch.save(checkpoint_state, filename)
    last_filename = os.path.join(output_dir, f"checkpoint_last.pt")
    shutil.copyfile(filename, last_filename)

def main():
    with open('/home/workspace/config_custom.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 시드 설정
    base_seed = config['system']['seed']
    set_seed(base_seed)
    print(f"Base random seed set to {base_seed}")

    device_id = 0
    device = f"cuda:{device_id}" if torch.cuda.is_available() and config['system']['device'] == 'auto' else "cpu"
    if device == "cpu": print("Error: DALI requires GPU."); return

    # AMP 사용 여부 결정    
    use_amp = config['training']['use_amp'] and device.startswith('cuda')
    print(f"Using device: {device}, AMP: {'ENABLED' if use_amp else 'DISABLED'}")

    # 모델 및 프로세서 로드
    model_id = config['model']['model_id']
    print(f"Loading model and processor for: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id).to(device)
    print("Model loaded.")

    # 옵티마이저 및 스케일러 설정
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    scaler = GradScaler(enabled=use_amp)
    
    global start_epoch
    start_epoch = 0
    # 이어서 학습할 체크포인트 경로 가져오기
    resume_checkpoint_path = config['training'].get('resume_from_checkpoint')

    # 체크포인트가 존재하면 모델, 옵티마이저, 스케일러 상태 복원
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

    # DALI 파이프라인을 위한 유효한 이미지 경로 및 텍스트 설명 준비
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
    save_interval = config['training'].get('save_checkpoint_interval', 1)  # 체크포인트 저장 간격

    for epoch in range(start_epoch, total_epochs):
        current_epoch = epoch + 1
        
        # 에포크별 시드 계산
        epoch_seed = base_seed + current_epoch - 1
        print(f"\n--- Starting Epoch {current_epoch}/{total_epochs} with DALI seed: {epoch_seed} ---")

        # DALI 파이프라인 생성
        '''
        batch_size: 배치 사이즈에 맞춰 메모리 버퍼를 미리 할당
        num_threads: 이미지 디코딩이나 증강 등을 몇 명이서 처리할지 세팅
        device_id: 몇 번 GPU를 사용할지 지정
        seed: 시드 지정(epoch_seed 사용)
        '''
        pipeline = DALITrainPipeline(
            image_paths=valid_paths, batch_size=config['training']['batch_size'],
            num_threads=config['system']['num_workers'], device_id=device_id,
            processor=processor,
            epoch_seed=epoch_seed # 에포크 시드 전달
        )
        pipeline.build()

        # 생성된 파이프라인으로 이터레이터(데이터로더) 생성
        '''
        pipelines: 파이프라인 리스트 (여기선 단일 GPU이므로 하나만 전달)
        reader_name: 파이프라인에서 정의한 이름표 지정(여러 파이프라인을 사용하더라도 이름표는 동일)
        last_batch_policy: 마지막 배치 처리 방식 지정
        (DROP: 남은 데이터가 배치 크기보다 작으면 버림,
         PARTIAL, FILL: 남은 데이터도 그대로 사용)
        auto_reset=True: 에포크가 끝나면 자동으로 이터레이터 초기화
        '''
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
            
        # 메모리 관리를 위해 사용 완료된 이터레이터와 파이프라인 삭제
        del dali_iterator
        del pipeline
            
    # 최종 모델 가중치 저장
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True) 
    model_name_safe = model_id.replace('/', '_')
    final_save_path = os.path.join(output_dir, f"finetuned_{model_name_safe}_final.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model state dict saved to {final_save_path}")

    print("Model fine-tuning complete.")

if __name__ == "__main__":
    main()

