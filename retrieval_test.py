import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torch.amp import autocast
import pickle

# --- DALI 라이브러리 임포트 ---
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

# --- embedding_DB.py와 동일한 DALI 파이프라인 클래스 ---
class DALIPipeline(Pipeline):
    def __init__(self, image_paths, batch_size, num_threads, device_id, processor):
        super(DALIPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        
        self.image_paths = image_paths
        
        # Processor로부터 모든 전처리 값을 동적으로 설정
        image_proc = processor.image_processor
        self.mean = image_proc.image_mean
        self.std = image_proc.image_std
        # --- 수정된 부분: 리사이즈와 크롭 사이즈를 별도로 가져옴 ---
        self.resize_size = image_proc.size['shortest_edge']
        self.crop_size = image_proc.crop_size['height']

    def define_graph(self):
        jpegs, labels = fn.readers.file(files=self.image_paths, name="file_reader")
        images = fn.decoders.image(jpegs, device="mixed")

        # --- 수정된 부분: CLIPProcessor의 전처리 순서를 정확히 모방 ---
        # 1. 리사이즈: 이미지의 짧은 쪽을 self.resize_size로 맞추고, 가로세로 비율 유지
        images = fn.resize(images, resize_shorter=self.resize_size)
        
        output_dtype = types.FLOAT16 if use_amp else types.FLOAT
        # 2. 중앙 크롭 & 정규화: 리사이즈된 이미지의 중앙에서 (crop_size, crop_size)로 잘라내고 정규화
        images = fn.crop_mirror_normalize(
            images,
            dtype=output_dtype,
            crop=(self.crop_size, self.crop_size), # 중앙 크롭 수행
            mean=[m * 255.0 for m in self.mean],  # 0-1 → 0-255 보정
            std=[s * 255.0 for s in self.std],
            output_layout="CHW"
        )
        return images, labels

def calculate_map(predictions, ground_truth):
    """mean Average Precision (mAP)을 계산하는 함수"""
    total_average_precision, valid_queries_count = 0.0, 0
    for query_id, retrieved_ids in predictions.items():
        if query_id not in ground_truth or not ground_truth[query_id]: continue
        valid_queries_count += 1
        relevant_ids = ground_truth[query_id]
        score, num_hits = 0.0, 0.0
        for i, p_id in enumerate(retrieved_ids):
            if p_id in relevant_ids:
                num_hits += 1.0
                precision_at_i = num_hits / (i + 1.0)
                score += precision_at_i
        average_precision = score / len(relevant_ids)
        total_average_precision += average_precision
    if valid_queries_count == 0: return 0.0
    return total_average_precision / valid_queries_count

def perform_retrieval():
    with open('/home/workspace/config_custom.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device_id = 0
    device = f"cuda:{device_id}" if torch.cuda.is_available() and config['system']['device'] == 'auto' else "cpu"
    if device == "cpu":
        print("Error: NVIDIA DALI requires a GPU to run.")
        return

    global use_amp
    use_amp = config['retrieval']['use_amp']
    print(f"Using device: {device}, AMP: {'ENABLED' if use_amp else 'DISABLED'}")

    print("Loading model, processor, and database...")
    processor = CLIPProcessor.from_pretrained(config['model']['model_id'], use_fast=True)
    model = CLIPModel.from_pretrained(config['model']['model_id']).to(device)
    finetuned_path = config['model'].get('finetuned_path')
    if finetuned_path and os.path.exists(finetuned_path):
        model.load_state_dict(torch.load(finetuned_path))
        print(f"Loaded fine-tuned weights from {finetuned_path}")
    model.eval()

    output_dir = config['paths']['output_dir']
    index = faiss.read_index(os.path.join(output_dir, "image_features.index"))
    with open(os.path.join(output_dir, "id_map.pkl"), 'rb') as f:
        id_map = pickle.load(f)
    print(f"Loading complete. Index contains {index.ntotal} vectors.")

    # --- 데이터 로딩 방식 변경: PyTorch -> DALI ---
    print("Preparing DALI pipeline for test images...")
    solution_df = pd.read_csv(config['paths']['solution_csv_path'])
    test_df = solution_df[solution_df['Usage'] != 'Ignored'].copy()
    
    # 유효한 test 이미지 경로와 ID 리스트 생성
    valid_paths, valid_ids = [], []
    for _, row in test_df.iterrows():
        img_path = os.path.join(config['paths']['test_image_dir'], row['id'][0], row['id'][1], row['id'][2], f"{row['id']}.jpg")
        if os.path.exists(img_path):
            valid_paths.append(img_path)
            valid_ids.append(row['id'])
            
    ground_truth = {row['id']: set(row['images'].split(' ')) for _, row in test_df.iterrows() if isinstance(row['images'], str)}

    # DALI 파이프라인 생성 (DB 생성 때와 동일한 로직)
    pipeline = DALIPipeline(
        image_paths=valid_paths, batch_size=config['retrieval']['batch_size'],
        num_threads=config['system']['num_workers'], device_id=device_id,
        processor=processor
    )
    pipeline.build()

    dali_iterator = DALIClassificationIterator(
        pipelines=[pipeline], reader_name="file_reader",
        last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True
    )
    print("DALI pipeline ready.")
    
    num_neighbors = min(config['retrieval']['num_neighbors'], index.ntotal)
    
    all_test_ids, all_retrieved_int_ids = [], []
    with torch.no_grad():
        for batch in tqdm(dali_iterator, desc="Performing retrieval with DALI"):
            images = batch[0]['data']
            labels = batch[0]['label'].squeeze(-1).long().tolist()
            test_ids = [valid_ids[i] for i in labels]
            
            with autocast(enabled=use_amp, device_type='cuda'):
                query_embeddings = model.get_image_features(pixel_values=images)
                query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)

            _, neighbor_int_ids = index.search(query_embeddings.cpu().float().numpy(), num_neighbors)
            
            all_test_ids.extend(test_ids)
            all_retrieved_int_ids.extend(neighbor_int_ids)

    predictions = {}
    for test_id, int_ids in zip(all_test_ids, all_retrieved_int_ids):
        uint_ids = np.array(int_ids, dtype=np.uint64)
        predictions[test_id] = [id_map[i.item()] for i in uint_ids if i.item() in id_map and i != -1]
        
    print("\nEvaluating performance...")
    map_score = calculate_map(predictions, ground_truth)
    print(f"mean Average Precision (mAP) @{num_neighbors}: {map_score:.4f}")

if __name__ == '__main__':
    perform_retrieval()

