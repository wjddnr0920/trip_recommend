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
    추론용 DALI 파이프라인.
    AutoProcessor로부터 전처리 값을 동적으로 읽어와 모든 모델에 자동 대응합니다.
    """
    def __init__(self, image_paths, batch_size, num_threads, device_id, processor):
        super(DALIInferencePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.image_paths = image_paths
        image_proc = processor.image_processor
        
        # processor가 CLIP이든 SigLIP이든, 알맞은 mean/std 값을 제공합니다.
        self.mean = image_proc.image_mean
        self.std = image_proc.image_std
        
        # crop_size(CLIP)와 size(SigLIP) 속성을 모두 안전하게 처리합니다.
        if hasattr(image_proc, 'crop_size') and isinstance(image_proc.crop_size, dict):
            self.crop_size = image_proc.crop_size['height']
        else:
            self.crop_size = image_proc.size['height']
        
        # 추론 시에는 리사이즈 크기도 필요합니다.
        self.resize_size = image_proc.size.get('shortest_edge', self.crop_size)

    def define_graph(self):
        jpegs, labels = fn.readers.file(files=self.image_paths, name="file_reader")
        images = fn.decoders.image(jpegs, device="mixed")
        # 비율 유지 리사이즈 -> 중앙 크롭의 정확한 추론 전처리
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
    """mean Average Precision (mAP)을 계산하는 함수"""
    total_average_precision, valid_queries_count = 0.0, 0
    for query_id, retrieved_ids in predictions.items():
        if query_id not in ground_truth or not ground_truth[query_id]: continue
        valid_queries_count += 1
        relevant_ids = ground_truth[query_id]
        score, num_hits = 0.0, 0.0
        for i, p_id in enumerate(retrieved_ids):
            if p_id in relevant_ids:
                num_hits += 1.0; precision_at_i = num_hits / (i + 1.0); score += precision_at_i
        average_precision = score / len(relevant_ids)
        total_average_precision += average_precision
    if valid_queries_count == 0: return 0.0
    return total_average_precision / valid_queries_count

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
                num_hits += 1.0; precision_at_i = num_hits / (i + 1.0); score += precision_at_i
        average_precision = score / len(relevant_ids)
        total_average_precision += average_precision
    if valid_queries_count == 0: return 0.0
    return total_average_precision / valid_queries_count

def perform_retrieval():
    # 사용자 환경에 맞는 config 파일 경로 사용
    with open('/home/workspace/config_custom.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device_id = 0
    device = f"cuda:{device_id}" if torch.cuda.is_available() and config['system']['device'] == 'auto' else "cpu"
    if device == "cpu":
        print("Error: NVIDIA DALI requires a GPU to run.")
        return

    use_amp = config['retrieval']['use_amp'] and device.startswith('cuda')
    print(f"Using device: {device}, AMP: {'ENABLED' if use_amp else 'DISABLED'}")

    print("Loading model, processor, and database...")
    processor = AutoProcessor.from_pretrained(config['model']['model_id'], use_fast=True)
    model = AutoModel.from_pretrained(config['model']['model_id']).to(device)
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
            
    ground_truth = {str(row['id']): set(row['images'].split(' ')) for _, row in test_df.iterrows() if isinstance(row['images'], str)}

    pipeline = DALIInferencePipeline(
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
