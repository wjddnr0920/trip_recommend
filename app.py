import os
import yaml
import torch
import torch.nn.functional as F
import faiss
import pickle
import numpy as np
from PIL import Image
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
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
    
    # 파인튜닝된 가중치 로드
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

    return {
        "config": config,
        "model": model,
        "processor": processor,
        "index": index,
        "id_map": id_map,
        "device": device
    }

# --- Lifespan (서버 시작/종료 시 실행) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 리소스 로드
    global resources
    resources = load_resources()
    yield
    # 종료 시 정리 (필요하면 추가)
    resources.clear()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# --- 엔드포인트 ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """메인 페이지 렌더링"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    """이미지를 업로드받아 검색 수행"""
    try:
        # 1. 업로드된 이미지 읽기
        contents = await file.read()
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 2. 모델 추론 (임베딩 추출)
        processor = resources["processor"]
        model = resources["model"]
        device = resources["device"]
        
        inputs = processor(images=query_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            embedding = F.normalize(features, p=2, dim=-1)
            query_vector = embedding.numpy()

        # 3. Faiss 검색
        index = resources["index"]
        id_map = resources["id_map"]
        
        k = 10 # 상위 10개 검색
        distances, indices = index.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            idx_item = int(idx)
            if idx_item in id_map:
                file_path = os.path.join("/home/workspace/data", id_map[idx_item])
                score = float(distances[0][i])
                # 파일 경로에서 파일명만 추출 (표시용)
                filename = os.path.basename(file_path)
                
                results.append({
                    "rank": i + 1,
                    "path": file_path, # 실제 전체 경로
                    "filename": filename,
                    "score": f"{score:.4f}"
                })

        return {"results": results}

    except Exception as e:
        print(f"Error during search: {e}")
        return {"error": str(e)}

@app.get("/image_proxy")
async def get_image(path: str):
    """로컬 경로의 이미지를 브라우저에 표시하기 위한 프록시"""
    if os.path.exists(path):
        return FileResponse(path)
    return HTTPException(status_code=404, detail="Image not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)