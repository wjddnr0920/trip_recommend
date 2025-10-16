import os
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

# --- 수정된 DALI 파이프라인 ---
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

def create_database():
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

    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model and processor...")
    processor = CLIPProcessor.from_pretrained(config['model']['model_id'], use_fast=True)
    model = CLIPModel.from_pretrained(config['model']['model_id']).to(device)
    finetuned_path = config['model'].get('finetuned_path')
    if finetuned_path and os.path.exists(finetuned_path):
        model.load_state_dict(torch.load(finetuned_path))
        print(f"Loaded fine-tuned model from: {finetuned_path}")
    model.eval()

    print("Preparing DALI pipeline...")
    df = pd.read_csv(config['paths']['index_csv_path'])
    valid_paths, valid_ids = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(config['paths']['index_image_dir'], row['id'][0], row['id'][1], row['id'][2], f"{row['id']}.jpg")
        if os.path.exists(img_path):
            valid_paths.append(img_path)
            valid_ids.append(row['id'])

    pipeline = DALIPipeline(
        image_paths=valid_paths, 
        batch_size=config['retrieval']['batch_size'],
        num_threads=config['system']['num_workers'], 
        device_id=device_id,
        processor=processor
    )
    pipeline.build()

    # --- 수정된 부분: DALI 이터레이터 설정 변경 ---
    dali_iterator = DALIClassificationIterator(
        pipelines=[pipeline],
        # 'size'를 직접 지정하는 대신, 'reader_name'을 알려주어 DALI가 자동으로 크기를 관리하도록 함
        reader_name="file_reader",
        last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True
    )
    print("DALI pipeline ready.")

    embedding_dim = model.config.projection_dim
    base_index = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIDMap(base_index)
    id_map = {} 
    
    with torch.no_grad():
        for batch in tqdm(dali_iterator, desc="Creating embeddings with DALI"):
            # DALI 이터레이터는 이제 [이미지 딕셔너리, 레이블 딕셔너리]를 반환
            images = batch[0]['data']
            labels = batch[0]['label'].squeeze(-1).long().tolist() # 레이블(인덱스) 추출
            
            # 레이블을 사용하여 원본 문자열 ID를 찾음
            str_image_ids = [valid_ids[i] for i in labels]
            
            with autocast(enabled=use_amp, device_type='cuda'):
                embeddings = model.get_image_features(pixel_values=images)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            int_image_ids = np.array([int(s_id, 16) for s_id in str_image_ids]).astype('uint64')
            index.add_with_ids(embeddings.cpu().float().numpy(), int_image_ids)
            
            for str_id, int_id in zip(str_image_ids, int_image_ids):
                id_map[int_id.item()] = str_id
                
    print(f"Successfully processed and indexed {index.ntotal} images.")
    
    faiss.write_index(index, os.path.join(output_dir, "image_features.index"))
    with open(os.path.join(output_dir, "id_map.pkl"), "wb") as f:
        pickle.dump(id_map, f)
    print(f"Faiss index and ID map saved to '{output_dir}'")

if __name__ == '__main__':
    create_database()

