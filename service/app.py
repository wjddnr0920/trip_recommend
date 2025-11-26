import os
import argparse
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
from typing import List

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoProcessor, AutoModel

# --- 전역 변수 ---
resources = {}

def load_resources():
    config_path = os.getenv("APP_CONFIG_PATH")
    
    if not config_path:
        # main 블록을 거치지 않고 uvicorn으로 직접 실행했을 경우를 대비한 에러 처리
        raise RuntimeError("Config path is missing. Please run with 'python app.py --config <path>'")
    
    print(f"Loading config from {config_path}...")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() and config['system']['device'] == 'auto' else "cpu"
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

    metadata_path = config['paths']['custom_metadata_csv']
    print(f"Loading metadata from {metadata_path}...")
    try:
        df = pd.read_csv(metadata_path)
        path_to_country = dict(zip(df['directory'].str.strip(), df['country'].str.strip()))
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

# --- 공통 검색 로직 함수 (리팩토링) ---
def process_search_results(query_vector, top_k, countries):
    index = resources["index"]
    id_map = resources["id_map"]
    path_to_country = resources["path_to_country"]
    
    target_countries = set(countries) if countries else {'korea', 'japan', 'china'}
    
    # 전체 검색 (필터링 대비)
    fetch_k = index.ntotal
    distances, indices = index.search(query_vector, fetch_k)

    results = []
    seen_filenames = set()

    for i, idx in enumerate(indices[0]):
        if idx == -1: continue
        idx_item = int(idx)
        
        if idx_item in id_map:
            rel_path = id_map[idx_item]
            file_path = os.path.join("/home/workspace/data", rel_path)
            
            img_country = path_to_country.get(rel_path.strip())
            if img_country and img_country not in target_countries:
                continue
            
            filename = os.path.basename(file_path)[:-4]
            if filename in seen_filenames:
                continue
            seen_filenames.add(filename)
            
            score = float(distances[0][i])
            
            results.append({
                "rank": len(results) + 1,
                "path": file_path,
                "filename": filename,
                "country": img_country,
                "score": f"{score:.4f}"
            })
            
            if len(results) >= top_k:
                break
    return results

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- 1. 이미지 검색 엔드포인트 ---
@app.post("/search_image")
async def search_by_image(
    file: UploadFile = File(...), 
    top_k: int = Form(10),
    countries: List[str] = Form([])
):
    try:
        contents = await file.read()
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")

        processor = resources["processor"]
        model = resources["model"]
        device = resources["device"]
        
        inputs = processor(images=query_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            embedding = F.normalize(features, p=2, dim=-1)
            query_vector = embedding.cpu().numpy()

        results = process_search_results(query_vector, top_k, countries)
        return {"results": results}

    except Exception as e:
        print(f"Error during image search: {e}")
        return {"error": str(e)}

# --- 2. 텍스트 검색 엔드포인트 ---
@app.post("/search_text")
async def search_by_text(
    text_query: str = Form(...),
    top_k: int = Form(10),
    countries: List[str] = Form([])
):
    try:
        processor = resources["processor"]
        model = resources["model"]
        device = resources["device"]
        
        # 텍스트 토크나이징 (tokenizer 명시적 호출)
        inputs = processor.tokenizer(
            text=[text_query],
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            # 텍스트 임베딩 추출
            features = model.get_text_features(**inputs)
            embedding = F.normalize(features, p=2, dim=-1)
            query_vector = embedding.cpu().numpy()

        results = process_search_results(query_vector, top_k, countries)
        return {"results": results}

    except Exception as e:
        print(f"Error during text search: {e}")
        return {"error": str(e)}

@app.get("/image_proxy")
async def get_image(path: str):
    if os.path.exists(path):
        return FileResponse(path)
    return HTTPException(status_code=404, detail="Image not found")

if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="FastAPI Image Search Server")
    
    # --- 수정된 부분: required=True로 설정하여 무조건 입력받게 함 ---
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file (REQUIRED)")
    
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    
    args = parser.parse_args()
    
    # 환경 변수에 설정 경로 저장
    os.environ["APP_CONFIG_PATH"] = args.config
    
    print(f"Starting server with config: {args.config}")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=True)