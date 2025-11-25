import os
import yaml
import torch
import torch.nn.functional as F
import faiss
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import AutoProcessor, AutoModel

# --- 전역 변수 (모델 및 리소스) ---
resources = {}

def load_resources():
    """설정, 모델, 인덱스를 로드합니다."""
    config_path = '/home/workspace/config_dataset.yaml'
    print(f"Loading config from {config_path}...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = "cpu" # CPU 강제 설정
    model_id = config['model']['model_id']
    output_dir = config['paths']['output_dir']

    print(f"Loading model: {model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id).to(device)
    
    finetuned_path = config['model'].get('finetuned_path')
    if finetuned_path and os.path.exists(finetuned_path):
        checkpoint = torch.load(finetuned_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print("Fine-tuned weights loaded.")
    model.eval()

    print("Loading DB...")
    index = faiss.read_index(os.path.join(output_dir, "image_features.index"))
    with open(os.path.join(output_dir, "id_map.pkl"), 'rb') as f:
        id_map = pickle.load(f)

    print(f"Loading metadata...")

    try:
        df = pd.read_csv(config['paths']['custom_metadata_csv'])
        # 'directory' 컬럼(파일 경로)을 Key로, 'country' 컬럼을 Value로 하는 딕셔너리 생성
        # 검색 속도를 위해 미리 Dict로 변환해둠
        path_to_country = dict(zip(df['directory'].str.strip(), df['country'].str.strip()))
        print(f"Loaded country info for {len(path_to_country)} images.")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        path_to_country = {}

    return {
        "config": config,
        "model": model,
        "processor": processor,
        "index": index,
        "id_map": id_map,
        "path_to_country": path_to_country,
        "device": device
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    global resources
    resources = load_resources()
    yield
    resources.clear()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search")
# top_k 파라미터를 Form으로 받음 (기본값 10)
async def search_image(file: UploadFile = File(...), top_k: int = Form(10), countries: list[str] = Form([])):
    """이미지를 업로드받아 중복 없는 검색 수행"""
    try:
        contents = await file.read()
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")

        processor = resources["processor"]
        model = resources["model"]
        device = resources["device"]
        path_to_country = resources["path_to_country"]
        
        # 선택된 국가가 없으면(빈 리스트) 모든 국가 검색으로 간주
        target_countries = set(countries) if countries else {'korea', 'japan', 'china'}

        # CPU 전용이므로 to(device)만 호출
        inputs = processor(images=query_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            embedding = F.normalize(features, p=2, dim=-1)
            query_vector = embedding.numpy()

        index = resources["index"]
        id_map = resources["id_map"]
        
        # --- 수정된 부분: 중복 제거 로직 ---
        target_k = top_k  # 최종적으로 보여줄 개수
        fetch_k = len(path_to_country)   # 필터링으로 인해 탈락할 후보를 고려하여 Faiss에서 전체 검색
        
        real_k = min(fetch_k, index.ntotal)
        distances, indices = index.search(query_vector, real_k)

        results = []
        seen_filenames = set() # 이미 결과에 담은 파일명을 기록할 집합

        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            idx_item = int(idx)
            
            if idx_item in id_map:
                metadata_path = id_map[idx_item]
                file_path = os.path.join("/home/workspace/data", metadata_path)

                # --- 국가 필터링 로직 ---
                # 메타데이터 딕셔너리에서 해당 이미지의 국가 정보를 찾음
                # 키가 없는 경우를 대비해 기본값 처리 필요할 수 있음
                img_country = path_to_country.get(metadata_path.strip())
                
                # 국가 정보가 있고, 사용자가 선택한 국가 목록에 없으면 건너뜀
                if img_country and img_country not in target_countries:
                    continue

                # --- 중복 제거 로직 ---                
                filename = os.path.basename(file_path)[:-4] # 경로에서 파일명만 추출 (예: 'A.jpg' -> 'A')
                
                # 이미 나온 파일명인지 확인
                if filename in seen_filenames:
                    continue # 중복이면 건너뜀

                # 중복이 아니면 추가
                seen_filenames.add(filename)

                score = float(distances[0][i])

                results.append({
                    "rank": len(results) + 1, # 현재 결과 리스트 길이를 기반으로 순위 매김
                    "path": file_path,
                    "filename": filename,
                    "country": img_country,
                    "score": f"{score:.4f}"
                })
                
                # 목표 개수를 채우면 중단
                if len(results) >= target_k:
                    break

        return {"results": results}

    except Exception as e:
        print(f"Error during search: {e}")
        return {"error": str(e)}

@app.get("/image_proxy")
async def get_image(path: str):
    if os.path.exists(path):
        return FileResponse(path)
    return HTTPException(status_code=404, detail="Image not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)