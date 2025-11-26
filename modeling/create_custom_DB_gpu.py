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

class DALIInferencePipeline(Pipeline):
    """
    추론용 DALI 파이프라인.
    AutoProcessor로부터 전처리 값을 동적으로 읽어와 모든 모델에 자동 대응합니다.
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
        # DALI가 반환하는 'labels'는 image_paths 리스트의 순차 인덱스(0, 1, 2...)가 됩니다.
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

def create_database(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device_id = 0
    device = f"cuda:{device_id}" if torch.cuda.is_available() and config['system']['device'] == 'auto' else "cpu"
    if device == "cpu":
        print("Error: NVIDIA DALI requires a GPU to run."); return

    use_amp = config['retrieval']['use_amp'] and device.startswith('cuda')
    
    model_id = config['model']['model_id']
    print(f"Loading model and processor for: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id).to(device)
    
    finetuned_path = config['model'].get('finetuned_path')
    if finetuned_path and os.path.exists(finetuned_path):
        model.load_state_dict(torch.load(finetuned_path))
    model.eval()

    print("Preparing DALI pipeline for custom dataset...")
    # --- 수정된 부분: 커스텀 config에서 경로와 컬럼 이름 읽기 ---
    df = pd.read_csv(config['paths']['custom_metadata_csv'])
    img_root = config['paths'].get('custom_image_root', '') # 기본값: 빈 문자열
    path_col = config['custom_dataset_columns']['image_path_column']
    id_col = config['custom_dataset_columns']['unique_id_column']

    valid_paths = []
    valid_string_ids = [] # 이 리스트가 [0, 1, 2...] 인덱스와 실제 문자열 ID를 매핑
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering custom dataset paths"):
        try:
            relative_path = str(row[path_col]).strip() # 공백 제거
            full_path = os.path.join(img_root, relative_path)
            
            if os.path.exists(full_path):
                valid_paths.append(full_path)
                valid_string_ids.append(str(row[id_col]).strip()) # 고유 ID로 사용할 문자열
            # else:
            #     print(f"Warning: Path not found, skipping: {full_path}")
        except Exception as e:
            print(f"Error processing row {row.name}: {e}")

    print(f"Found {len(valid_paths)} valid custom images.")
    if not valid_paths:
        print("Error: No valid images found. Check config paths and CSV file.")
        return

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
    
    embedding_dim = model.config.projection_dim if hasattr(model.config, "projection_dim") else model.config.text_config.projection_size
    base_index = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIDMap(base_index)
    
    # id_map은 { Faiss ID (int) -> String ID }를 저장
    id_map = {} 
    
    with torch.no_grad():
        for batch in tqdm(dali_iterator, desc="Creating embeddings"):
            images, labels = batch[0]['data'], batch[0]['label'].squeeze(-1).long().tolist()
            
            # --- 수정된 부분: 새로운 ID 매핑 로직 ---
            # 'labels'는 valid_paths 리스트의 순차 인덱스 (0, 1, 2, ...)
            # 이 순차 인덱스를 Faiss ID로 사용
            int_image_ids = np.array(labels, dtype=np.uint64)
            
            # 이 인덱스를 사용해 실제 문자열 ID 조회
            str_image_ids = [valid_string_ids[i] for i in labels]
            
            with autocast(enabled=use_amp, device_type='cuda'):
                embeddings = model.get_image_features(pixel_values=images)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            index.add_with_ids(embeddings.cpu().float().numpy(), int_image_ids)
            
            # id_map에 {순차 인덱스(int) : 실제 문자열 ID} 저장
            for int_id, str_id in zip(int_image_ids, str_image_ids):
                id_map[int_id.item()] = str_id
                
    print(f"Successfully processed and indexed {index.ntotal} images.")
    
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "image_features.index"))
    with open(os.path.join(output_dir, "id_map.pkl"), "wb") as f:
        pickle.dump(id_map, f)
    print(f"Faiss index and ID map saved to '{output_dir}'")

if __name__ == '__main__':
    # 사용할 config 파일 경로를 지정
    create_database(config_path='/home/workspace/config_dataset.yaml')
