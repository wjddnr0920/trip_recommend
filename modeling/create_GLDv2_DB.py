import os
import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import faiss
import numpy as np
from transformers import AutoProcessor, AutoModel
from torch.amp import autocast
import pickle

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

class DALIPipeline(Pipeline):
    """
    DB 생성을 위한 DALI 파이프라인
    image_paths: 이미지 파일 경로 리스트
    batch_size: 배치 크기
    num_threads: CPU 스레드 개수
    device_id: GPU 장치 ID
    processor: HuggingFace AutoProcessor 객체
    seed는 고정
    """
    def __init__(self, image_paths, batch_size, num_threads, device_id, processor):
        super(DALIPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
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
        
        # DB를 생성할 때 리사이즈 크기도 필요
        # 속성에 'shortest_edge'가 있으면 그것을 사용, 없으면 crop_size 사용
        self.resize_size = image_proc.size.get('shortest_edge', self.crop_size)

    def define_graph(self):
        '''
        fn.readers.file: 이미지를 읽는 DALI 함수
        files: 이미지 파일 경로 리스트
        name: 이름표 지정 --> DALIClassificationIterator에서 참조(데이터의 개수 파악)
        output
        jpegs: 이미지 바이트(압축되어있음)
        labels: 정수 레이블 --> 이미지에 맞는 이미지 파일명(str_image_ids)을 찾아내는 인덱스로 사용(create_database 함수에 활용)
        '''
        jpegs, labels = fn.readers.file(files=self.image_paths, name="file_reader")
        images = fn.decoders.image(jpegs, device="mixed")

        '''
        fn.resize(images, resize_shorter=224): 이미지의 짧은 변이 224가 되도록 비율을 유지하며 리사이즈
        600X400 이미지가 들어왔을 때 목표가 224X224라면
        1. 짧은 변(400)이 224가 되도록 리사이즈
        2. 비율을 유지하므로 336X224가 됨
        3. 이후 중앙 crop을 통해 224X224로 오려냄 
        '''
        images = fn.resize(images, resize_shorter=self.resize_size)  # 비율 유지 리사이즈 -> 중앙 crop
        output_dtype = types.FLOAT
        # 이미지 정규화 후 DALI의 범위(0~255)로 변환
        images = fn.crop_mirror_normalize(
            images, dtype=output_dtype, crop=(self.crop_size, self.crop_size),
            mean=[m * 255.0 for m in self.mean],
            std=[s * 255.0 for s in self.std],
            output_layout="CHW"
        )
        return images, labels

def create_database():
    with open('/home/workspace/config_custom.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device_id = 0
    device = f"cuda:{device_id}" if torch.cuda.is_available() and config['system']['device'] == 'auto' else "cpu"
    if device == "cpu":
        print("Error: NVIDIA DALI requires a GPU to run.")
        return

    # AMP 사용 여부 결정   
    use_amp = config['retrieval']['use_amp'] and device.startswith('cuda')
    print(f"Using device: {device}, AMP: {'ENABLED' if use_amp else 'DISABLED'}")

    # 모델 및 프로세서 로드
    model_id = config['model']['model_id']
    print(f"Loading model and processor for: {model_id}")

    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(config['model']['model_id'], use_fast=True)
    model = AutoModel.from_pretrained(config['model']['model_id']).to(device)

    # 파인튜닝된 모델이 있다면 불러오기
    finetuned_path = config['model'].get('finetuned_path')
    if finetuned_path and os.path.exists(finetuned_path):
        model.load_state_dict(torch.load(finetuned_path))
        print(f"Loaded fine-tuned model from: {finetuned_path}")

    model.eval()

    # DALI 파이프라인을 위한 유효한 이미지 경로 준비
    print("Preparing DALI pipeline...")
    df = pd.read_csv(config['paths']['index_csv_path'])
    valid_paths, valid_ids = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(config['paths']['index_image_dir'], str(row['id'])[0], str(row['id'])[1], str(row['id'])[2], f"{row['id']}.jpg")
        if os.path.exists(img_path):
            valid_paths.append(img_path)
            valid_ids.append(str(row['id']))

    # DALI 파이프라인 생성
    '''
    batch_size: 배치 사이즈에 맞춰 메모리 버퍼를 미리 할당
    num_threads: 이미지 디코딩이나 증강 등을 몇 명이서 처리할지 세팅
    device_id: 몇 번 GPU를 사용할지 지정
    '''
    pipeline = DALIPipeline(
        image_paths=valid_paths, 
        batch_size=config['retrieval']['batch_size'],
        num_threads=config['system']['num_workers'], 
        device_id=device_id,
        processor=processor
    )
    pipeline.build()

    # 생성된 파이프라인으로 이터레이터(데이터로더) 생성
    '''
    pipelines: 파이프라인 리스트 (여기선 단일 GPU이므로 하나만 전달)
    reader_name: 파이프라인에서 정의한 이름표 지정(여러 파이프라인을 사용하더라도 이름표는 동일)
    last_batch_policy: 마지막 배치 처리 방식 지정
    (PARTIAL: 남은 데이터도 그대로 사용)
    auto_reset=True: 싸이클 한 번이 끝나면 자동으로 이터레이터 초기화
    '''
    dali_iterator = DALIClassificationIterator(
        pipelines=[pipeline], reader_name="file_reader",
        last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True
    )
    print("DALI pipeline ready.")

    # Faiss DB를 만들기 위해선 임베딩의 길이(차원)을 알아야 함
    # CLIP 모델의 경우
    if hasattr(model.config, "projection_dim"):
        embedding_dim = model.config.projection_dim
        print(f"Embedding dimension (projection_dim) detected: {embedding_dim}")
    # SigLIP 시리즈 모델의 경우
    elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "projection_size"):
        embedding_dim = model.config.text_config.projection_size
        print(f"Embedding dimension (text_config.projection_size) detected: {embedding_dim}")
    else:
        # 안전하게 에러 발생
        raise AttributeError(f"Could not automatically determine embedding dimension for model {model_id}.")
    
    '''
    faiss.IndexFlatIP(embedding_dim)
    Index: Faiss에서 검색을 담당하는 객체
    Flat: 벡터를 압축하지 않고 원본 그대로 저장(모든 데이터를 하나도 빠짐없이 비교)
    IP: 내적(Inner Product)을 사용  --> 이 때 벡터의 길이를 1로 정규화하면 코사인 유사도와 동일
    embedding_dim: 벡터의 차원 수(벡터 하나가 몇 개의 숫자로 이루어져 있는지)

    faiss.IndexIDMap
    IndexFlatIP의 한계: 벡터에 대한 고유한 ID를 직접 지정할 수 없음
    IndexIDMap은 IndexFlatIP를 래핑하여 각 벡터에 대해 고유한 ID(정수만 가능)를 지정할 수 있게 함
    '''
    base_index = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIDMap(base_index)
    id_map = {} 
    
    with torch.no_grad():
        for batch in tqdm(dali_iterator, desc="Creating embeddings with DALI"):
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
            str_image_ids = [valid_ids[i] for i in labels]  # 이미지에 해당하는 이미지 파일명
            
            # 임베딩 생성
            with autocast(enabled=use_amp, device_type='cuda'):
                embeddings = model.get_image_features(pixel_values=images)  # 이미지 representation 추출
                '''
                embeddings의 shape: [배치 크기, 임베딩 차원]
                normalize: 임베딩 벡터를 정규화하여 크기가 1이 되도록 함
                p=2: L2 정규화
                dim=-1: 마지막 차원(임베딩)을 기준으로 길이를 계산
                이미지 벡터가 개별적으로 길이가 1이 됨
                '''
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            '''
            이미지 파일명 처리
            GLDv2 데이터셋의 경우 이미지 파일명이 7a4a4a5d7775508d.jpg처럼 16진수 문자열임
            Faiss IndexIDMap은 정수형 ID만 지원
            따라서 16진수 문자열을 정수형으로 변환하여 저장
            '''
            # 파일명 처리
            # int(s_id, 16): 문자열을 16진수로 해석하여 정수로 변환
            # 오버플로우를 대비해 'uint64' 타입으로 변환
            int_image_ids = np.array([int(s_id, 16) for s_id in str_image_ids]).astype('uint64')
            # (임베딩, 정수형 이미지 ID) 쌍을 Faiss 인덱스에 추가
            # faiss는 CPU 메모리에서 작동, faiss의 표준 규격은 float32, faiss는 numpy 배열을 사용
            index.add_with_ids(embeddings.cpu().float().numpy(), int_image_ids)
            
            # ID 맵 저장
            # 검색할 때 "정수 ID -> 실제 이미지 파일명"을 알기 위해 딕셔너리에 저장
            for str_id, int_id in zip(str_image_ids, int_image_ids):
                id_map[int_id.item()] = str_id
                
    print(f"Successfully processed and indexed {index.ntotal} images.")
    
    # Faiss 인덱스 파일(.index)과 ID 매핑 파일(.pkl) 저장
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "image_features.index"))
    with open(os.path.join(output_dir, "id_map.pkl"), "wb") as f:
        pickle.dump(id_map, f)
    print(f"Faiss index and ID map saved to '{output_dir}'")

if __name__ == '__main__':
    create_database()
