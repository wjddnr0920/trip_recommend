import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# DALI 라이브러리 임포트
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

class DALIInferencePipeline(Pipeline):
    """
    추론용 DALI 파이프라인
    DB 생성 파이프라인과 동일
    """
    def __init__(self, image_paths, batch_size, num_threads, device_id, processor):
        super(DALIInferencePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.image_paths = image_paths

        image_proc = processor.image_processor
        self.mean = image_proc.image_mean
        self.std = image_proc.image_std
        
        if hasattr(image_proc, 'crop_size') and isinstance(image_proc.crop_size, dict):
            self.crop_size = image_proc.crop_size['height']
        else:
            self.crop_size = image_proc.size['height']
        
        self.resize_size = image_proc.size.get('shortest_edge', self.crop_size)

    def define_graph(self):
        jpegs, labels = fn.readers.file(files=self.image_paths, name="file_reader")
        images = fn.decoders.image(jpegs, device="mixed")

        images = fn.resize(images, resize_shorter=self.resize_size)
        output_dtype = types.FLOAT
        
        images = fn.crop_mirror_normalize(
            images, dtype=output_dtype, crop=(self.crop_size, self.crop_size),
            mean=[m * 255.0 for m in self.mean],
            std=[s * 255.0 for s in self.std],
            output_layout="CHW"
        )
        return images, labels

def calculate_map(predictions, ground_truth):
    """
    mean Average Precision (mAP)을 계산하는 함수
    predictions: {'쿼리 ID': [검색된 관련 이미지 ID 리스트]}  --> 순서가 중요하기 때문에 리스트로 저장함
    ground_truth: {'쿼리 ID': set(관련 이미지 ID들)}  --> 순서는 상관없고 포함 여부만 중요하기 때문에 집합으로 저장함
    """
    total_average_precision, valid_queries_count = 0.0, 0
    for query_id, retrieved_ids in predictions.items():
        # 쿼리 이미지가 ground_truth에 없거나 관련 이미지가 아예 없다면 건너뜀
        if query_id not in ground_truth or not ground_truth[query_id]:
            continue

        valid_queries_count += 1
        relevant_ids = ground_truth[query_id]  # GT 관련 이미지 리스트
        score, num_hits = 0.0, 0.0

        for i, p_id in enumerate(retrieved_ids):  # 예측 관련 이미지 리스트 순회
            if p_id in relevant_ids:  # 예측 관련 이미지가 정답이었다면
                num_hits += 1.0
                precision_at_i = num_hits / (i + 1.0)  # Precision 계산
                score += precision_at_i
        average_precision = score / len(relevant_ids)  # AP 계산
        total_average_precision += average_precision  # 모든 쿼리의 AP 합산
    if valid_queries_count == 0: return 0.0
    return total_average_precision / valid_queries_count  # mAP return

def perform_retrieval():
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
    print("Loading model, processor, and database...")
    processor = AutoProcessor.from_pretrained(config['model']['model_id'], use_fast=True)
    model = AutoModel.from_pretrained(config['model']['model_id']).to(device)

    # 파인튜닝된 모델이 있다면 불러오기
    finetuned_path = config['model'].get('finetuned_path')
    if finetuned_path and os.path.exists(finetuned_path):
        model.load_state_dict(torch.load(finetuned_path))
        print(f"Loaded fine-tuned weights from {finetuned_path}")

    model.eval()

    # Faiss 인덱스 및 ID 맵 로드
    output_dir = config['paths']['output_dir']
    index = faiss.read_index(os.path.join(output_dir, "image_features.index"))
    with open(os.path.join(output_dir, "id_map.pkl"), 'rb') as f:
        id_map = pickle.load(f)
    print(f"Loading complete. Index contains {index.ntotal} vectors.")

    # DALI 파이프라인을 위한 유효한 이미지 경로 준비
    print("Preparing DALI pipeline for test images...")
    solution_df = pd.read_csv(config['paths']['solution_csv_path'])
    # 'Ignored'를 제외한 모든 쿼리 사용 (Public + Private)
    test_df = solution_df[solution_df['Usage'] != 'Ignored'].copy()
    
    valid_paths, valid_ids = [], []
    for _, row in test_df.iterrows():
        img_path = os.path.join(config['paths']['test_image_dir'], str(row['id'])[0], str(row['id'])[1], str(row['id'])[2], f"{row['id']}.jpg")
        if os.path.exists(img_path):
            valid_paths.append(img_path)
            valid_ids.append(str(row['id']))
    
    # GT 딕셔너리 생성 {'쿼리 ID': set(관련 이미지 ID들)}
    ground_truth = {str(row['id']): set(row['images'].split(' ')) for _, row in test_df.iterrows() if isinstance(row['images'], str)}

    # DALI 파이프라인 생성
    '''
    batch_size: 배치 사이즈에 맞춰 메모리 버퍼를 미리 할당
    num_threads: 이미지 디코딩이나 증강 등을 몇 명이서 처리할지 세팅
    device_id: 몇 번 GPU를 사용할지 지정
    '''
    pipeline = DALIInferencePipeline(
        image_paths=valid_paths, batch_size=config['retrieval']['batch_size'],
        num_threads=config['system']['num_workers'], device_id=device_id,
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

    # 검색할 이웃 수
    num_neighbors = min(config['retrieval']['num_neighbors'], index.ntotal)

    all_test_ids, all_retrieved_int_ids = [], []
    with torch.no_grad():
        for batch in tqdm(dali_iterator, desc="Performing retrieval with DALI"):
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
            test_ids = [valid_ids[i] for i in labels]  # 이미지에 해당하는 이미지 파일명
            
            with autocast(enabled=use_amp, device_type='cuda'):
                query_embeddings = model.get_image_features(pixel_values=images)  # 이미지 representation 추출
                '''
                query_embeddings의 shape: [배치 크기, 임베딩 차원]
                normalize: 임베딩 벡터를 정규화하여 크기가 1이 되도록 함
                p=2: L2 정규화
                dim=-1: 마지막 차원(임베딩)을 기준으로 길이를 계산
                이미지 벡터가 개별적으로 길이가 1이 됨
                '''
                query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)

            '''
            index.search(query, k)의 return 값: (distances, indices)
            distances: 쿼리 벡터와 검색된 벡터 간의 거리(유사도)
            indices: 검색된 벡터들이 DB에 저장될 때 부여받은 고유 번호(여기선 정수 ID)
            '''
            # Faiss DB에서 쿼리 임베딩과 가장 유사한 벡터 num_neighbors개를 검색
            _, neighbor_int_ids = index.search(query_embeddings.cpu().float().numpy(), num_neighbors)
            all_test_ids.extend(test_ids)
            all_retrieved_int_ids.extend(neighbor_int_ids)

    predictions = {}
    '''
    zip(all_test_ids, all_retrieved_int_ids)
    all_test_ids: 쿼리 이미지 파일명 리스트 --> ['img1', 'img2', ...]
    all_retrieved_int_ids: 쿼리 이미지와 유사한 이미지들의 ID 리스트 --> [[105, 33, ...], [12, 78, ...], ...]

    Faiss는 내부적으로 int64를 사용하지만, ID 맵은 uint64를 사용하므로 변환 필요

    [id_map[i.item()] for i in uint_ids if i.item() in id_map and i != -1]
    1. for i in uint_ids: 검색된 ID를 하나씩 순회
    2. if i.item() in id_map and i != -1:
        - i != -1: Faiss가 "결과 없음"을 의미하는 -1을 반환했다면 제외
        - i.item() in id_map: ID 맵에 해당 ID가 존재하는지 확인(.item()은 넘파이나 텐서를 파이썬 기본 타입으로 변환)
    3. id_map[i.item()]: 유효한 ID에 대한 실제 이미지 파일명
    '''
    for test_id, int_ids in zip(all_test_ids, all_retrieved_int_ids):
        uint_ids = np.array(int_ids, dtype=np.uint64)
        predictions[test_id] = [id_map[i.item()] for i in uint_ids if i.item() in id_map and i != -1]
        
    print("\nEvaluating performance...")
    map_score = calculate_map(predictions, ground_truth)
    print(f"mean Average Precision (mAP) @{num_neighbors}: {map_score:.4f}")

if __name__ == '__main__':
    perform_retrieval()
