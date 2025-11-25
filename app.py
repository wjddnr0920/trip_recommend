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

    return {
        "config": config,
        "model": model,
        "processor": processor,
        "index": index,
        "id_map": id_map,
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
async def search_image(file: UploadFile = File(...), top_k: int = Form(10)):
    """이미지를 업로드받아 중복 없는 검색 수행"""
    try:
        contents = await file.read()
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")

        processor = resources["processor"]
        model = resources["model"]
        device = resources["device"]
        
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
        fetch_k = max(50, target_k*3)   # 중복을 고려해 넉넉하게 가져올 후보 개수(최소 50개, 요청이 많으면 요청 * 3배)
        
        real_k = min(fetch_k, index.ntotal)
        distances, indices = index.search(query_vector, real_k)

        results = []
        seen_filenames = set() # 이미 결과에 담은 파일명을 기록할 집합

        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            idx_item = int(idx)
            
            if idx_item in id_map:
                file_path = os.path.join("/home/workspace/data", id_map[idx_item])
                score = float(distances[0][i])
                filename = os.path.basename(file_path)[:-4] # 경로에서 파일명만 추출 (예: 'A.jpg' -> 'A')
                
                # 이미 나온 파일명인지 확인
                if filename in seen_filenames:
                    continue # 중복이면 건너뜀
                
                # 중복이 아니면 추가
                seen_filenames.add(filename)
                results.append({
                    "rank": len(results) + 1, # 현재 결과 리스트 길이를 기반으로 순위 매김
                    "path": file_path,
                    "filename": filename,
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