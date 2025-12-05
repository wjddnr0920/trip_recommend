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

# 모델, DB, processor 등을 저장할 전역 변수
resources = {}

def load_resources():
    '''
    서버가 시작할 때 단 한 번만 실행되는 함수
    모델, 프로세서, 검색 DB 등을 로드 후 반환
    '''
    config_path = os.getenv("APP_CONFIG_PATH")
    
    if not config_path:
        raise RuntimeError("Config path is missing. Please run with 'python app.py --config <path>'")
    
    print(f"Loading config from {config_path}...")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 모델 및 프로세서 로드
    device = "cuda" if torch.cuda.is_available() and config['system']['device'] == 'auto' else "cpu"
    model_id = config['model']['model_id']
    output_dir = config['paths']['output_dir']

    print(f"Loading model: {model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id).to(device)
    
    # 파인튜닝된 모델이 있다면 불러오기
    finetuned_path = config['model'].get('finetuned_path')
    if finetuned_path and os.path.exists(finetuned_path):
        checkpoint = torch.load(finetuned_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print("Fine-tuned weights loaded.")

    model.eval()

    # 검색 DB 로드
    print("Loading DB...")
    index = faiss.read_index(os.path.join(output_dir, "image_features.index"))
    with open(os.path.join(output_dir, "id_map.pkl"), 'rb') as f:
        id_map = pickle.load(f)

    # 국가 필터링을 위한 메타데이터 로드
    metadata_path = config['paths']['custom_metadata_csv']
    print(f"Loading metadata from {metadata_path}...")
    try:
        df = pd.read_csv(metadata_path)
        # 'directory' 열과 'country' 열로 key-value 매핑
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
    '''
    Liftspan 이벤트
    1. 전체 서버의 생명주기를 관리
    2. 코드 구조
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            ...
            yield
            ...
    3. @asynccontextmanager 데코레이터를 사용하여 비동기 컨텍스트 매니저를 정의(원하는 타이밍에 리소스를 할당하고 제공하는 역할)
    4. yield를 기준으로
        yield 앞에 있는 코드는 서버가 시작될 때 실행,
        yield 뒤에 있는 코드는 서버가 종료될 때 실행
    '''
    global resources
    resources = load_resources()
    yield
    resources.clear()

# FastAPI 앱 및 템플릿 설정
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    '''
    사용자가 서버에 접속했을 때 index.html 파일을 화면에 보여주는(랜더링) 역할

    @app.get("/", response_class=HTMLResponse)
    메인 페이지 접속 시(GET 요청) 웹 페이지 화면(HTML)을 보여줌
    FastAPI는 기본적으로 데이터를 JSON({"key": "value"}) 형식으로 돌려주기 때문에 HTMLResponse으로 명시

    async def read_root(request: Request):
    사용자의 요청을 서버로 보낼 때 request 객체로 보냄
    Jinja2 템플릿을 사용할 때는 반드시 request 객체를 템플릿에게 넘겨줘야함

    return templates.TemplateResponse("index.html", {"request": request})
    index.html 템플릿을 꺼내서 사용자의 요청을 처리해서 다시 웹 페이지에 전달
    '''
    return templates.TemplateResponse("index.html", {"request": request})

# 이미지&텍스트 Retrieval 함수
def process_search_results(query_vector, top_k, countries):
    # Faiss 인덱스 및 매핑 정보 로드
    index = resources["index"]
    id_map = resources["id_map"]
    # 국가 필터링을 위한 메타데이터 로드
    path_to_country = resources["path_to_country"]
    
    target_countries = set(countries) if countries else {'korea', 'japan', 'china'}
    
    # DB 전체 데이터 검색
    fetch_k = index.ntotal
    distances, indices = index.search(query_vector, fetch_k)

    results = []
    seen_filenames = set()

    '''
    index.search가 반환하는 indices의 shape: (쿼리 개수, 검색 개수)
    사용자가 1개의 이미지나 텍스트를 입력하기 때문에 항상 (1, fetch_k)
    따라서 indices[0]을 사용
    '''
    # 상위 랭크부터 순회
    for i, idx in enumerate(indices[0]):
        # 결과 없음"을 의미하는 -1이라면 skip
        if idx == -1: continue
        idx_item = int(idx)
        
        if idx_item in id_map:
            # IDMap으로 ID -> 이미지 경로 변환
            rel_path = id_map[idx_item]
            file_path = os.path.join(resources["config"]['paths']['custom_image_root'], rel_path)
            
            # [필터링 1] 국가 필터링
            # 사용자가 설정한 국가 목록(target_countries)에 없으면 skip
            img_country = path_to_country.get(rel_path.strip())
            if img_country and img_country not in target_countries:
                continue
            
            # [필터링 2] 중복 이미지 제거
            # 이미 검색된 이미지라면 skip 
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

# 이미지 검색 엔드포인트(/search_image)
@app.post("/search_image")
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    countries: List[str] = Form([])
):
    '''
    웹 페이지(프론트엔드)에서 이미지 검색 요청을 보냈을 때
    서버(백엔드)가 이 요청을 어떻게 받아서 처리할지 정의

    @app.post('/search_image)
    http://localhost:8000/search_image 주소로 POST 요청을 받을 시 search_by_image 함수 실행
    데이터 구조는 index.html에서 확인 가능

    프론트엔드는 FormData로 모든 데이터를 하나의 상자로 담아서 서버로 보냄
    file: UploadFile = File(...)
    - file: POST 요청 시 서버가 받는 데이터 중 하나
    - UploadFile: file 변수에 파일 데이터가 들어온다고 알려줌, Python 파일 객체처럼 read() 사용 가능
    - File(...): 이미지를 다뤄야하므로 file 폼 데이터를 File()로 처리 , 사용자가 안 보냈으면 에러 발생

    top_k: int = Form(10)
    - top_k: POST 요청 시 서버가 받는 데이터 중 하나
    - int: top_k의 타입 지정
    - Form(10): 단순 텍스트이므로 top_k 폼 데이터를 Form()로 처리, 사용자가 안 보냈으면 default로 10 사용

    countries: List[str] = Form([])
    - countries: POST 요청 시 서버가 받는 데이터 중 하나
    - List[str]: countries는 문자열들의 리스트임을 지정(데이터가 여러 개일 수 있기 때문)
    - Form([]): 단순 텍스트이므로 countries 폼 데이터를 Form()로 처리, 사용자가 안 보냈으면 빈 리스트로 처리
    '''
    try:
        # 업로드된 파일 읽기
        contents = await file.read()
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")

        processor = resources["processor"]
        model = resources["model"]
        device = resources["device"]
        
        inputs = processor(images=query_image, return_tensors="pt").to(device)
        
        # 이미지 -> 벡터(임베딩) 변환
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            embedding = F.normalize(features, p=2, dim=-1)
            query_vector = embedding.cpu().numpy()

        # Retrieval 수행
        results = process_search_results(query_vector, top_k, countries)
        return {"results": results}

    except Exception as e:
        print(f"Error during image search: {e}")
        return {"error": str(e)}

# 텍스트 검색 엔드포인트
@app.post("/search_text")
async def search_by_text(
    text_query: str = Form(...),
    top_k: int = Form(10),
    countries: List[str] = Form([])
):
    '''
    웹 페이지(프론트엔드)에서 텍스트 검색 요청을 보냈을 때
    서버(백엔드)가 이 요청을 어떻게 받아서 처리할지 정의

    @app.post('/search_text)
    http://localhost:8000/search_text 주소로 POST 요청을 받을 시 search_by_text 함수 실행
    데이터 구조는 index.html에서 확인 가능

    text_query: str = Form(...)
    - text_query: POST 요청 시 서버가 받는 데이터의 이름
    - str: text_query는 문자열이라고 명시
    - Form(...): 단순 텍스트이므로 text_query 폼 데이터를 Form()로 처리, 사용자가 안 보냈으면 에러 발생
    '''
    try:
        processor = resources["processor"]
        model = resources["model"]
        device = resources["device"]
        
        # 텍스트 토크나이징(SigLIP2 세팅)
        inputs = processor.tokenizer(
            text=[text_query],
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True
        ).to(device)
        
        # 텍스트 -> 벡터(임베딩) 변환
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            embedding = F.normalize(features, p=2, dim=-1)
            query_vector = embedding.cpu().numpy()

        # Retrieval 수행
        results = process_search_results(query_vector, top_k, countries)
        return {"results": results}

    except Exception as e:
        print(f"Error during text search: {e}")
        return {"error": str(e)}

# 웹페이지로 이미지 전달
@app.get("/image_proxy")
async def get_image(path: str):
    '''
    서버에서 웹 페이지로 유사도 검색 결과를 보낼 때 이미지가 아닌 파일 경로를 보냄
    브라우저에서 서버의 파일 경로에 접근해야하지만 보안상 접근이 불가능
    따라서 서버에게 이미지를 보내달라고 요청해야함

    @app.get("/image_proxy")
    /image_proxy에서 이미지 데이터를 요청하는 GET 방식 사용

    async def get_image(path: str):
    브라우저가 요청할 때 주소 뒤에 ?path=/home/workspace/data/cat.jpg 처럼 물음표 뒤에 경로를 붙여서 보냄(?는 구분자 문법)
    FastAPI는 주소의 path와 get_image 함수의 path를 보고 /home/workspace/data/cat.jpg를 path 변수에 저장

    return FileResponse(path)
    이미지 데이터 전송
    '''
    if os.path.exists(path):
        return FileResponse(path)
    return HTTPException(status_code=404, detail="Image not found")

if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="FastAPI Image Search Server")
    
    # argparse를 통해 커맨드라인에서 인자 입력
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file (REQUIRED)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    
    args = parser.parse_args()
    
    # 입력받은 config 경로를 환경 변수에 저장해 load_resources 함수가 읽을 수 있게 설정
    os.environ["APP_CONFIG_PATH"] = args.config
    
    print(f"Starting server with config: {args.config}")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=True)